"""Protected pre-user checkpoint, owned by the existing single-flight supervisor.

No public administrative routes, second scheduler, generated seed, or LRU files.
Only exact native template/tokenizer input enters the saved state. Any cache error falls
back to ordinary inference; compatibility never relies on model geometry alone.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class SinglePrefixCheckpoint:
    FORMAT=1
    MAX_BYTES=2*1024**3
    def __init__(self,profile,post,event,*,manifest_file='manifest.json'):
        self.profile=profile;self.post=post;self.event=event;self.lock=threading.Lock()
        self.root=Path(profile['prefix_checkpoint_dir'])
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if self.root.is_symlink() or self.root.stat().st_uid!=os.getuid():
            raise ValueError('Prefix directory must be private and owned')
        os.chmod(self.root,0o700)
        self.identity=digest({k:v for k,v in profile.items() if k not in
            ('_profile','deployment_state','fit_note','front_host','front_port','idle_seconds',
             'load_timeout_seconds','request_timeout_seconds','body_limit_bytes')})
        if manifest_file not in ('manifest.json','base-1.json','base-2.json'):
            raise ValueError('Invalid managed manifest')
        self.manifest_file=manifest_file
        self.manifest=None;self.record=None;self.pid=None;self.known_sources={}
        self._target_payload=None;self._target_tokens=None
        self.value={'state':'missing','reason':'no_checkpoint','tokens':0,'bytes':0,
            'protected_from_lru':True,'pre_user_only':True,'updated_at':None}
        self._read()

    def status(self):
        with self.lock:return dict(self.value)

    def _state(self,state,reason,**fields):
        with self.lock:
            changed=(state,reason)!=(self.value['state'],self.value['reason'])
            self.value.update(state=state,reason=reason,**fields)
        if changed:self.event('prefix_checkpoint',phase=state,reason=reason)

    def _read(self):
        was_known=(self.record is not None or self.manifest is not None
            or self.value.get('reason')=='checkpoint_manifest_missing')
        self.manifest=None;self.record=None
        try:
            path=self.root/self.manifest_file
            if not path.exists():
                self._state('invalid' if was_known else 'missing',
                    'checkpoint_manifest_missing' if was_known else 'no_checkpoint',tokens=0,bytes=0)
                return
            if path.is_symlink() or path.stat().st_size>4*1024*1024:
                raise ValueError('manifest')
            m=json.loads(path.read_text())
            tokens=m['tokens'];filename=m['filename']
            if (not isinstance(tokens,list) or not 16<=len(tokens)<=65536
                    or any(type(x) is not int or x<0 for x in tokens)
                    or type(m.get('bytes')) is not int or not 0<=m['bytes']<=self.MAX_BYTES
                    or filename!='prefix-'+digest(tokens)+'.bin'):
                raise ValueError('manifest')
            self.record=m
            if m.get('format')!=self.FORMAT or m.get('identity')!=self.identity:
                self._state('invalid','runtime_or_model_changed',tokens=len(tokens));return
            file=self.root/filename
            if (file.is_symlink() or not file.is_file() or file.stat().st_size!=m['bytes']
                    or file_digest(file)!=m['sha256']):raise ValueError('state')
            self.manifest=m
            self._state('saved','validated_on_disk',tokens=len(tokens),bytes=m['bytes'],
                updated_at=m['created_at'])
        except (OSError,ValueError,KeyError,TypeError,RecursionError):
            self.manifest=None;self._state('invalid','checkpoint_damaged')

    def on_load(self,pid):
        """Called only under the request gate after the native child is ready."""
        if self.pid==pid:return
        self.pid=pid;self.known_sources.clear()
        # A supervisor may outlive several child models. Validate again even if
        # the SSD artifact was healthy when this supervisor first started.
        self._read()
        if self.manifest:
            try:
                result=self.post('/slots/0?action=restore',{'filename':self.manifest['filename']})
                if result.get('n_restored')!=len(self.manifest['tokens']):
                    raise ValueError('Incomplete prefix restore')
                self._state('restored','disk_imported',restored_at=time.time())
            except (OSError,ValueError,RuntimeError):
                self._erase();self.manifest=None
                self._state('invalid','native_restore_failed')

    def _erase(self):
        try:self.post('/slots/0?action=erase',{})
        except (OSError,RuntimeError,ValueError):pass

    def _prefix(self,payload):
        messages=payload.get('messages')
        # Auxiliary summaries/titles have different instructions and must never
        # replace the tool-aware Hermes base with a tiny unrelated checkpoint.
        if not isinstance(messages,list) or not payload.get('tools'):return None,None
        leading=[]
        for msg in messages:
            if not isinstance(msg,dict) or msg.get('role') not in ('system','developer'):break
            if not isinstance(msg.get('content'),str):return None,None
            item=deepcopy(msg)
            # Hermes adds date/session/runtime hints after this final delimiter.
            # They remain in the original inference payload, but excluding this
            # small variable tail avoids rewriting the SSD base for date changes.
            head,marker,_=item['content'].rpartition('\n\nConversation started: ')
            if marker:item['content']=head
            leading.append(item)
        if not leading or len(leading)==len(messages):return None,None
        # Model-specific template options may affect the prefix. Sampling does not.
        shape={k:deepcopy(payload[k]) for k in ('tools','tool_choice','parallel_tool_calls',
            'chat_template','chat_template_kwargs','reasoning_format','reasoning_effort') if k in payload}
        shape['messages']=deepcopy(leading)
        source=digest(shape)
        if source in self.known_sources:return source,self.known_sources[source]
        partial=dict(shape,add_generation_prompt=False)
        full=deepcopy(payload)
        rendered=self.post('/apply-template',partial)['prompt']
        actual=self.post('/apply-template',full)['prompt']
        a=self.post('/tokenize',{'content':rendered,'add_special':True,'parse_special':True})['tokens']
        b=self.post('/tokenize',{'content':actual,'add_special':True,'parse_special':True})['tokens']
        self._target_payload=payload;self._target_tokens=b
        # The intersection excludes any invented generation header and real user.
        n=0
        for x,y in zip(a,b):
            if type(x) is not int or x!=y:break
            n+=1
        tokens=a[:n]
        if not 16<=len(tokens)<=min(65536,self.profile['context']//2):return source,None
        self.known_sources[source]=tokens
        if len(self.known_sources)>8:self.known_sources.pop(next(iter(self.known_sources)))
        return source,tokens

    def full_tokens(self,payload):
        if payload is self._target_payload:return self._target_tokens
        actual=self.post('/apply-template',deepcopy(payload))['prompt']
        return self.post('/tokenize',{'content':actual,'add_special':True,'parse_special':True})['tokens']

    def prepare(self,payload,*,prepared=None,cleanup=True):
        """Build only when prefix changes; never reset an advanced live conversation."""
        mutated=False
        try:
            source,tokens=prepared if prepared is not None else self._prefix(payload)
            if tokens is None:return
            if self.manifest and tokens==self.manifest['tokens']:
                self._state('ready','current_prefix_valid');return
            self._state('rebuilding','prefix_changed' if self.manifest else 'initial_prefix',
                tokens=len(tokens))
            # This native build samples one discarded token even with n_predict=0.
            # Its sampled token has not been decoded/appended to the saved KV.
            # The save-count invariant below verifies the exact pre-user boundary.
            mutated=True
            result=self.post('/completion',{'prompt':tokens,'n_predict':0,'id_slot':0,
                'cache_prompt':True,'stream':False})
            filename='prefix-'+digest(tokens)+'.bin'
            saved=self.post('/slots/0?action=save',{'filename':filename})
            if saved.get('n_saved')!=len(tokens):raise ValueError('Incomplete prefix save')
            file=self.root/filename
            if file.is_symlink() or not file.is_file():raise ValueError('Invalid native checkpoint')
            if file.stat().st_size>self.MAX_BYTES:
                # Never publish an oversized artifact. An existing other base
                # remains protected; this native request can proceed normally.
                if not self.record or filename!=self.record['filename']:file.unlink()
                raise ValueError('Protected base exceeds SSD allowance')
            os.chmod(file,0o600)
            with file.open('rb') as f:os.fsync(f.fileno())
            m={'format':self.FORMAT,'identity':self.identity,'source':source,'tokens':tokens,
                'filename':filename,'bytes':file.stat().st_size,'sha256':file_digest(file),
                'created_at':time.time()}
            temp=self.root/(self.manifest_file+'.pending')
            with temp.open('w') as f:
                os.chmod(temp,0o600);json.dump(m,f,separators=(',',':'));f.flush();os.fsync(f.fileno())
            os.replace(temp,self.root/self.manifest_file)
            fd=os.open(self.root,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            self.manifest=m;self.record=m
            if cleanup:
                for old in self.root.glob('prefix-*.bin'):
                    if old.name!=filename:old.unlink()
            self._state('ready','checkpoint_updated',tokens=len(tokens),bytes=m['bytes'],
                updated_at=m['created_at'])
        except (OSError,ValueError,RuntimeError,KeyError,TypeError):
            if mutated:self._erase()
            self._state('invalid','cache_prepare_failed')


class PrefixCheckpoint:
    """Two independent protected bases, using native transactional RAM selection.

    The opt-in native marker is verified by the supervisor before any request.
    Legacy deployments keep their original single-base behavior until upgraded.
    """
    def __new__(cls,profile,post,event):
        if not profile.get('prefix_dual_cache',False):
            return SinglePrefixCheckpoint(profile,post,event)
        return super().__new__(cls)

    def __init__(self,profile,post,event):
        self.profile=profile;self.post=post;self.event=event
        self.root=Path(profile['prefix_checkpoint_dir']);self.pid=None
        self.selected=None;self.selection=None;self.native_selection=None
        self.lock=threading.RLock();self.last_used={}
        self.entries={key:SinglePrefixCheckpoint(profile,post,event,manifest_file=key+'.json')
            for key in ('base-1','base-2')}
        self.parser=self.entries['base-1']
        self._migrate_legacy()
        self._read_catalog()
        for key,entry in self.entries.items():
            if self.last_used.get(key,0)>0 and not (self.root/(key+'.json')).exists():
                entry._state('invalid','checkpoint_manifest_missing',tokens=0,bytes=0)

    def _migrate_legacy(self):
        if any((self.root/(key+'.json')).exists() for key in self.entries):return
        legacy=SinglePrefixCheckpoint(self.profile,self.post,self.event)
        if legacy.manifest:
            os.replace(self.root/'manifest.json',self.root/'base-1.json')
            self.entries['base-1']._read()

    def _read_catalog(self):
        try:
            path=self.root/'catalog.json'
            if path.is_symlink() or path.stat().st_size>16384:raise ValueError('catalog')
            value=json.loads(path.read_text())
            if value.get('format')!=2:raise ValueError('catalog')
            self.selected=value.get('selected') if value.get('selected') in self.entries else None
            for key,stamp in (value.get('last_used') or {}).items():
                if key in self.entries and type(stamp) in (int,float):self.last_used[key]=stamp
        except (OSError,ValueError,TypeError):pass

    def _catalog(self):
        path=self.root/'catalog.pending'
        with path.open('w') as f:
            os.chmod(path,0o600)
            json.dump({'format':2,'selected':self.selected,'last_used':self.last_used},f)
            f.flush();os.fsync(f.fileno())
        os.replace(path,self.root/'catalog.json')

    def status(self):
        with self.lock:
            bases=[]
            for key,entry in self.entries.items():
                value=entry.status()
                bases.append(dict(value,id=key,selected=key==self.selected))
            good={'saved','restored','ready'}
            ready=sum(base['state'] in good for base in bases)
            current=next((b for b in bases if b['id']==self.selected),bases[0])
            aggregate=dict(current)
            aggregate.update(bases=bases,selected_base=self.selected,count_ready=ready,count_total=2,
                storage_limit_bytes=2*SinglePrefixCheckpoint.MAX_BYTES,
                storage_used_bytes=sum((e.record or {}).get('bytes',0) for e in self.entries.values()),
                base_limit_bytes=SinglePrefixCheckpoint.MAX_BYTES,
                protected_from_lru=True,pre_user_only=True)
            if ready and aggregate['state'] not in good:
                aggregate.update(state='ready',reason='another_base_available')
            if self.selection:
                aggregate['selection_mode']=self.selection['mode']
                aggregate['selection_ambiguous']=self.selection['ambiguous']
            if self.native_selection:aggregate['native_selection']=dict(self.native_selection)
            return aggregate

    def _native(self,entry,target):
        m=entry.manifest
        result=self.post('/slots/0?action=restore',{'filename':m['filename'],
            'restore_if_better':True,'expected_prefix_tokens':m['tokens'],'target_tokens':target})
        source=result.get('restore_source')
        if source not in ('active','ram','disk'):
            raise RuntimeError('Native transactional prefix selection unsupported')
        self.native_selection={key:result[key] for key in
            ('restore_source','n_reusable','n_restored','preserve_failed','restore_reason') if key in result}
        if source=='disk' and result.get('n_restored')!=len(m['tokens']):
            raise ValueError('Incomplete base restore')
        return result

    def on_load(self,pid):
        if self.pid==pid:return
        self.pid=pid;self.parser.known_sources.clear()
        # Import both under the existing request gate. Native preservation keeps
        # the first in the bounded RAM pool when the second enters the slot.
        for entry in self.entries.values():
            entry._read()
            if not entry.manifest:continue
            try:
                result=self._native(entry,entry.manifest['tokens'])
                if result.get('preserve_failed'):
                    entry._state('saved','native_preservation_unavailable')
                else:entry._state('restored','disk_imported',restored_at=time.time())
            except (OSError,ValueError,RuntimeError):
                entry.manifest=None;entry._state('invalid','native_restore_failed')

    def _collect_unreferenced(self):
        # Read committed pointers, including after a partial metadata failure.
        # An unreadable pointer means ownership is uncertain: retain its files.
        referenced=set()
        for name in ('manifest.json','base-1.json','base-2.json'):
            path=self.root/name
            if not path.exists():continue
            try:
                if path.is_symlink() or path.stat().st_size>4*1024*1024:return
                value=json.loads(path.read_text());filename=value['filename']
                if filename!='prefix-'+digest(value['tokens'])+'.bin':return
                referenced.add(filename)
            except (OSError,ValueError,KeyError,TypeError):return
        for file in self.root.glob('prefix-*.bin'):
            if file.name not in referenced:file.unlink()

    @staticmethod
    def _has_media(payload):
        # /apply-template + /tokenize cannot represent MTMD image/audio chunks.
        # Leave those histories to the real native multimodal cache selector;
        # restoring a text-only estimate could discard a warmer visual state.
        for message in payload.get('messages',[]):
            content=message.get('content') if isinstance(message,dict) else None
            if isinstance(content,list) and any(not isinstance(part,dict)
                    or part.get('type') not in ('text','input_text') for part in content):return True
        return False

    def prepare(self,payload):
        from prefix_selection import choose_base
        try:
            if self._has_media(payload):
                # Guard before rendering: apply-template itself can fetch image
                # URLs/files, and its text tokens do not encode real MTMD chunks.
                # Process media only once, in the actual inference request.
                self.selection=None
                self.native_selection={'restore_source':'native_multimodal',
                    'restore_reason':'real_media_chunks_required'}
                return
            try:source,tokens=self.parser._prefix(payload)
            except (OSError,ValueError,RuntimeError,KeyError,TypeError):
                # Rendering/tokenizing a request is not an integrity test of
                # either saved artifact. Preserve their independently verified
                # state and let the actual inference endpoint handle the input.
                self.native_selection={'restore_source':'unavailable',
                    'restore_reason':'native_prefix_preview_unavailable'}
                return
            if tokens is None:return
            records=[]
            for key,entry in self.entries.items():
                m=entry.record
                if m:records.append({'id':key,'tokens':m['tokens'],'occupied':True,
                    'runtime':m['identity'],'source_identity':m.get('source'),
                    'valid':entry.manifest is not None,'last_used':self.last_used.get(key,0)})
            selection=choose_base(tokens,records,runtime=self.parser.identity,source_identity=source)
            self.selection=selection
            key=selection['id'];entry=self.entries[key]
            if selection['mode']=='exact':
                try:target=self.parser.full_tokens(payload)
                except (OSError,ValueError,RuntimeError,KeyError,TypeError):
                    entry._state('saved','native_target_preview_unavailable')
                    self.native_selection={'restore_source':'unavailable',
                        'restore_reason':'native_target_preview_unavailable'}
                    return
                result=self._native(entry,target)
                if result.get('preserve_failed'):
                    entry._state('saved','native_preservation_unavailable')
                else:entry._state('ready','current_prefix_valid')
            else:
                # Ordinary native completion can retain only a matching part
                # of its current slot. Protect its exact endpoint in RAM before
                # building a changed base; never save a live user state to SSD.
                preserved=self.post('/slots/0?action=restore',{'filename':'prefix-prepare.bin',
                    'restore_if_better':True,'preserve_active_only':True})
                if preserved.get('restore_source')!='active':
                    raise RuntimeError('Native active-state preservation unsupported')
                if preserved.get('preserve_failed'):
                    entry._state('pending','active_preservation_unavailable')
                    self.native_selection={k:preserved[k] for k in
                        ('restore_source','n_restored','preserve_failed','restore_reason') if k in preserved}
                    return
                # Native prefill handles a changed prefix; only its chosen entry
                # is published. The other manifest/state are never overwritten.
                entry.prepare(payload,prepared=(source,tokens),cleanup=False)
                if not entry.manifest or entry.manifest['tokens']!=tokens:return
                self.native_selection={'restore_source':'prepared',
                    'restore_reason':'pre_user_base_built','prefix_tokens':len(tokens)}
            changed=self.selected!=key
            self.selected=key;self.last_used[key]=time.time()
            if changed or selection['mode']!='exact':self._catalog()
        except (OSError,ValueError,RuntimeError,KeyError,TypeError):
            # Do not erase a live advanced slot after a selection failure: the
            # native operation is transactional and normal inference stays safe.
            if self.selected:self.entries[self.selected]._state('invalid','cache_prepare_failed')
        finally:
            self.parser._target_payload=None;self.parser._target_tokens=None
            try:self._collect_unreferenced()
            except OSError:pass
