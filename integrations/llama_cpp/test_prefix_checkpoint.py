"""Controlled persistence and exact pre-user boundary; never invokes a model."""
import json
from pathlib import Path
import tempfile
import unittest
from prefix_checkpoint import PrefixCheckpoint


class Native:
    def __init__(self,root):self.root=Path(root);self.calls=[];self.tokens=[];self.fail=False
    def post(self,path,p):
        self.calls.append((path,p))
        if self.fail:raise RuntimeError('controlled native failure')
        if path=='/apply-template':
            text='TOOLS:'+json.dumps(p.get('tools',[]),sort_keys=True)+'\n'
            text+=''.join('<'+m['role']+'>'+m['content']+'<end>' for m in p['messages'])
            return {'prompt':text+('<assistant>' if p.get('add_generation_prompt',True) else '')}
        if path=='/tokenize':return {'tokens':[ord(x) for x in p['content']]}
        if path=='/completion':
            assert p['n_predict']==0
            self.tokens=p['prompt'];return {'tokens_predicted':0}
        if path.endswith('=save'):
            (self.root/p['filename']).write_text(json.dumps(self.tokens))
            return {'n_saved':len(self.tokens)}
        if path.endswith('=restore'):
            self.tokens=json.loads((self.root/p['filename']).read_text())
            return {'n_restored':len(self.tokens)}
        if path.endswith('=erase'):self.tokens=[];return {}
        raise AssertionError(path)


class Tests(unittest.TestCase):
    def test_variable_hermes_tail_stays_in_request_but_outside_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            native=Native(tmp)
            app=PrefixCheckpoint({'prefix_checkpoint_dir':tmp,'context':102400},native.post,
                lambda *a,**kw:None)
            core='Stable instructions and up-to-date memory.'
            p={'messages':[{'role':'system','content':core+'\n\nConversation started: Monday\nSession: A\nCWD: /one'},
                {'role':'user','content':'private first question'}],'tools':[{'name':'read'}]}
            original=json.dumps(p,sort_keys=True)
            app.prepare(p);tokens=app.manifest['tokens'];source=app.manifest['source']
            self.assertEqual(json.dumps(p,sort_keys=True),original)
            p['messages'][0]['content']=core+'\n\nConversation started: Tuesday\nSession: B\nCWD: /two'
            p['messages'][1]['content']='different private question'
            app.prepare(p)
            self.assertEqual((app.manifest['tokens'],app.manifest['source']),(tokens,source))
            self.assertNotIn('Conversation started',''.join(map(chr,tokens)))
            self.assertIn('/two',p['messages'][0]['content'])

    def test_prefix_restart_memory_tools_and_corruption_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile={'prefix_checkpoint_dir':tmp,'context':102400,'model':'verified-A'}
            native=Native(tmp);events=[]
            app=PrefixCheckpoint(profile,native.post,lambda *a,**kw:events.append(kw))
            p={'messages':[{'role':'system','content':'Core instructions and live memory.'},
                {'role':'user','content':'private greeting NEVER CACHE THIS'}],
                'tools':[{'name':'research','description':'Search and read evidence.'}]}
            app.on_load(1);app.prepare(p)
            self.assertEqual(app.status()['state'],'ready')
            manifest=json.loads((Path(tmp)/'manifest.json').read_text())
            decoded=''.join(map(chr,manifest['tokens']))
            self.assertNotIn('private greeting',decoded)
            self.assertNotIn('<user>',decoded)
            self.assertNotIn('<assistant>',decoded)
            self.assertEqual(manifest['tokens'],native.tokens)
            # Different question and later long conversation do not reset live slot.
            native.tokens+=[999]
            p['messages'][1]['content']='An unrelated question'
            app.prepare(p)
            self.assertEqual(native.tokens[-1],999)
            count=len(native.calls)
            app.prepare(p)
            self.assertEqual(len(native.calls),count)
            restored=PrefixCheckpoint(profile,native.post,lambda *a,**kw:None)
            restored.on_load(2)
            self.assertEqual(restored.status()['state'],'restored')
            self.assertEqual(native.tokens,manifest['tokens'])
            p['messages'][0]['content']+=' Updated memory.'
            restored.prepare(p)
            self.assertNotEqual(restored.manifest['tokens'],manifest['tokens'])
            p['tools'].append({'name':'pdf','description':'Write and read documents.'})
            restored.prepare(p)
            self.assertIn('pdf',''.join(map(chr,restored.manifest['tokens'])))
            self.assertEqual(len(list(Path(tmp).glob('prefix-*.bin'))),1)
            self.assertEqual((Path(tmp)/'manifest.json').stat().st_mode&0o777,0o600)
            state=Path(tmp)/restored.manifest['filename'];state.write_bytes(b'corrupt')
            native.calls.clear();restored.on_load(3)
            self.assertEqual(restored.status()['reason'],'checkpoint_damaged')
            self.assertEqual(native.calls,[])
            damaged=PrefixCheckpoint(profile,native.post,lambda *a,**kw:None)
            self.assertEqual(damaged.status()['reason'],'checkpoint_damaged')
            native.calls.clear();damaged.on_load(3)
            self.assertEqual(native.calls,[])
            damaged.prepare(p)
            incompatible=PrefixCheckpoint(dict(profile,model='verified-B'),native.post,lambda *a,**kw:None)
            self.assertEqual(incompatible.status()['reason'],'runtime_or_model_changed')
            native.calls.clear();incompatible.on_load(4)
            self.assertEqual(native.calls,[])

    def test_no_core_and_native_failure_never_block_normal_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            native=Native(tmp);events=[]
            app=PrefixCheckpoint({'prefix_checkpoint_dir':tmp,'context':102400},native.post,
                lambda *a,**kw:events.append(kw))
            app.prepare({'messages':[{'role':'user','content':'private'}]})
            self.assertEqual(native.calls,[])
            native.fail=True
            app.prepare({'messages':[{'role':'system','content':'Stable core instructions.'},
                {'role':'user','content':'private'}],'tools':[{'name':'fixture'}]})
            self.assertEqual(app.status()['state'],'invalid')
            self.assertFalse((Path(tmp)/'manifest.json').exists())


if __name__=='__main__':unittest.main()
