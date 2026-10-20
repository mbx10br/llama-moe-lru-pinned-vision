"""Controlled wrapper acceptance; model-facing selection is tested natively too."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from prefix_checkpoint import PrefixCheckpoint, SinglePrefixCheckpoint
from prefix_selection import common_prefix
from test_prefix_checkpoint import Native


class TransactionalNative(Native):
    def __init__(self,root):
        super().__init__(root);self.pool=[];self.keep=True
    def post(self,path,p):
        if path.endswith('=restore') and p.get('restore_if_better'):
            self.calls.append((path,copy.deepcopy(p)))
            if p.get('preserve_active_only'):
                if self.keep and self.tokens:self.pool.append(self.tokens[:])
                return {'restore_source':'active','n_restored':len(self.tokens),
                    'preserve_failed':not self.keep}
            target=p['target_tokens'];expected=p['expected_prefix_tokens']
            assert target[:len(expected)]==expected
            candidates=[('active',self.tokens)]+[('ram',x) for x in self.pool]
            candidates.append(('disk',json.loads((self.root/p['filename']).read_text())))
            source,tokens=max(candidates,key=lambda x:common_prefix(x[1],target))
            if source!='active':
                if not self.keep:return {'restore_source':'active','preserve_failed':True,
                    'n_restored':len(self.tokens),'n_reusable':common_prefix(self.tokens,target)}
                if self.tokens:self.pool.append(self.tokens[:])
                self.tokens=tokens[:]
            return {'restore_source':source,'n_restored':len(self.tokens),
                'n_reusable':common_prefix(self.tokens,target),'preserve_failed':False}
        if path=='/completion' and self.tokens:self.pool.append(self.tokens[:])
        return super().post(path,p)


def payload(name):
    return {'messages':[{'role':'system','content':(name+' instructions. ')*15},
        {'role':'user','content':'PRIVATE USER NEVER STORE'}],
        'tools':[{'name':name,'description':(name+' tool usage. ')*20}]}


class DualTests(unittest.TestCase):
    def setup_app(self,tmp):
        native=TransactionalNative(tmp)
        profile={'prefix_dual_cache':True,'prefix_checkpoint_dir':tmp,'context':102400}
        return native,profile,PrefixCheckpoint(profile,native.post,lambda *a,**k:None)

    def test_two_profiles_restart_and_advanced_conversation_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,p,app=self.setup_app(tmp);a=payload('WEB');b=payload('TELEGRAM')
            app.on_load(1);app.prepare(a);app.prepare(b)
            self.assertEqual(app.status()['count_ready'],2)
            original={f.name:f.read_bytes() for f in Path(tmp).glob('prefix-*.bin')}
            for entry in app.entries.values():
                self.assertNotIn('PRIVATE USER',''.join(map(chr,entry.manifest['tokens'])))
                self.assertNotIn('<assistant>',''.join(map(chr,entry.manifest['tokens'])))
            n.tokens=[];n.pool=[];n.calls=[]
            app=PrefixCheckpoint(p,n.post,lambda *a,**k:None);app.on_load(2)
            self.assertEqual(app.status()['count_ready'],2)
            for body in (a,b,a):app.prepare(body)
            self.assertFalse(any(path=='/completion' for path,_ in n.calls))
            app.prepare(a)
            full=app.parser.full_tokens(a);n.tokens=full[:-1]
            advanced=n.tokens[:]
            app.prepare(b);app.prepare(a)
            self.assertEqual(app.native_selection['restore_source'],'ram')
            self.assertEqual(n.tokens,advanced)
            app.prepare(a)
            self.assertEqual(app.native_selection['restore_source'],'active')
            self.assertEqual(original,{f.name:f.read_bytes() for f in Path(tmp).glob('prefix-*.bin')})

    def test_front_edit_updates_closest_only_and_corruption_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,p,app=self.setup_app(tmp);a=payload('WEB');b=payload('TELEGRAM')
            app.prepare(a);app.prepare(b)
            b_manifest=(Path(tmp)/'base-2.json').read_bytes()
            b_file=Path(tmp)/app.entries['base-2'].manifest['filename'];b_bytes=b_file.read_bytes()
            a['tools'][0]['description']='NEW '+a['tools'][0]['description']
            app.prepare(a)
            self.assertEqual(app.selected,'base-1')
            self.assertEqual((Path(tmp)/'base-2.json').read_bytes(),b_manifest)
            self.assertEqual(b_file.read_bytes(),b_bytes)
            self.assertEqual(len(list(Path(tmp).glob('prefix-*.bin'))),2)
            (Path(tmp)/app.entries['base-1'].manifest['filename']).write_bytes(b'CORRUPT')
            n.tokens=[];n.pool=[];n.calls=[]
            app=PrefixCheckpoint(p,n.post,lambda *a,**k:None);app.on_load(3)
            self.assertEqual(app.status()['count_ready'],1)
            app.prepare(b)
            self.assertEqual(app.selection['mode'],'exact')
            self.assertEqual(app.selected,'base-2')
            self.assertFalse(any(path=='/completion' for path,_ in n.calls))

    def test_legacy_rename_does_not_copy_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,p,_=self.setup_app(tmp)
            old=SinglePrefixCheckpoint(p,n.post,lambda *a,**k:None);old.prepare(payload('WEB'))
            f=Path(tmp)/old.manifest['filename'];inode=f.stat().st_ino;content=f.read_bytes()
            manifest=(Path(tmp)/'manifest.json').read_bytes()
            app=PrefixCheckpoint(p,n.post,lambda *a,**k:None)
            self.assertEqual((Path(tmp)/'base-1.json').read_bytes(),manifest)
            self.assertFalse((Path(tmp)/'manifest.json').exists())
            self.assertEqual((f.stat().st_ino,f.read_bytes()),(inode,content))
            self.assertEqual(app.entries['base-1'].manifest,old.manifest)

    def test_hot_visual_history_uses_real_native_media_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,p,app=self.setup_app(tmp);a=payload('WEB');app.prepare(a)
            a['messages'][-1]['content']=[{'type':'text','text':'Inspect image'},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,fixture'}}]
            original=copy.deepcopy(a);n.tokens+=[12345];advanced=n.tokens[:];n.calls=[]
            app.prepare(a)
            self.assertEqual(n.tokens,advanced)
            self.assertEqual(a,original)
            self.assertEqual(n.calls,[])
            self.assertEqual(app.status()['count_ready'],1)
            self.assertEqual(app.native_selection['restore_source'],'native_multimodal')
            cold=PrefixCheckpoint(p,n.post,lambda *a,**k:None);n.calls=[]
            cold.prepare(a)
            self.assertEqual(n.calls,[])  # no duplicate image URL/file retrieval
            self.assertEqual(n.tokens,advanced)

    def test_bounded_storage_and_native_preservation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,p,app=self.setup_app(tmp);a=payload('WEB');b=payload('TELEGRAM')
            app.prepare(a);app.prepare(b)
            unchanged=(Path(tmp)/'base-2.json').read_bytes()
            changed=copy.deepcopy(a);changed['messages'][0]['content']+='small update'
            with patch.object(SinglePrefixCheckpoint,'MAX_BYTES',1):app.prepare(changed)
            self.assertEqual((Path(tmp)/'base-2.json').read_bytes(),unchanged)
            self.assertEqual(len(list(Path(tmp).glob('prefix-*.bin'))),2)
            n.tokens=[999999];n.pool=[];n.keep=False
            before=len([path for path,_ in n.calls if path=='/completion'])
            app.prepare(changed)
            self.assertEqual(len([path for path,_ in n.calls if path=='/completion']),before)
            self.assertEqual(n.tokens,[999999])
            app.prepare(b)
            self.assertEqual(n.tokens,[999999])
            self.assertTrue(app.native_selection['preserve_failed'])
            self.assertFalse(any(path.endswith('=erase') for path,_ in n.calls[-1:]))

    def test_preview_failure_is_not_checkpoint_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,p,app=self.setup_app(tmp);a=payload('WEB');app.prepare(a)
            before=(Path(tmp)/'base-1.json').read_bytes()
            n.fail=True;app.prepare(a)
            self.assertEqual(app.status()['count_ready'],1)
            self.assertEqual(app.entries['base-1'].status()['reason'],'native_target_preview_unavailable')
            a['messages'][0]['content']+='new core requiring render'
            app.prepare(a)
            self.assertEqual(app.status()['count_ready'],1)
            self.assertEqual((Path(tmp)/'base-1.json').read_bytes(),before)
            self.assertEqual(app.native_selection['restore_reason'],'native_prefix_preview_unavailable')

    def test_disappeared_manifest_is_not_reported_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            n,p,app=self.setup_app(tmp);app.prepare(payload('WEB'));app.prepare(payload('TELEGRAM'))
            (Path(tmp)/'base-1.json').unlink()
            fresh=PrefixCheckpoint(p,n.post,lambda *a,**k:None)
            first=fresh.entries['base-1'].status()
            self.assertEqual((first['state'],first['reason']),('invalid','checkpoint_manifest_missing'))
            self.assertEqual(fresh.status()['count_ready'],1)
            fresh.on_load(3)
            self.assertEqual(fresh.entries['base-1'].status()['state'],'invalid')
            app.on_load(3)
            self.assertEqual(app.entries['base-1'].status()['state'],'invalid')
            self.assertEqual(app.status()['count_ready'],1)


if __name__=='__main__':unittest.main()
