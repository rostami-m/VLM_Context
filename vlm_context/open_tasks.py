"""Official-style VQA normalization and leave-one-annotator-out scoring.
Vendored from the separately audited historical implementation; no model imports.
"""
import re

_ARTICLES = {"a", "an", "the"}

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id've": "i'd've", "i'dve": "i'd've", "im": "i'm", "ive": "i've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}

_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}

_VQA_PUNCTUATION = (';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
                    '_', '-', '>', '<', '@', '`', ',', '?', '!')

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")

_COMMA_BETWEEN_DIGITS = re.compile(r"\d,\d")

def _normalize_vqa_punctuation(text: str) -> str:
    output = text
    for punctuation in _VQA_PUNCTUATION:
        if (f"{punctuation} " in text or f" {punctuation}" in text
                or _COMMA_BETWEEN_DIGITS.search(text)):
            output = output.replace(punctuation, "")
        else:
            output = output.replace(punctuation, " ")
    return _PERIOD_STRIP.sub("", output)

def vqa_norm(s):
    words = _normalize_vqa_punctuation(str(s).lower().replace("\n", " ").replace("\t", " ")).split()
    normalized = []
    for word in words:
        word = _NUMBERS.get(word, word)
        if word not in _ARTICLES:
            normalized.append(_CONTRACTIONS.get(word, word))
    return " ".join(normalized)

def vqa_score(pred, answers):
    """Official-style VQAv2 leave-one-annotator-out soft accuracy."""
    p = vqa_norm(pred)
    gold = [vqa_norm(a) for a in answers]
    if len(gold) <= 1:
        return float(bool(gold) and p == gold[0])
    per_annotator = []
    for index in range(len(gold)):
        matches = sum(p == answer for other, answer in enumerate(gold) if other != index)
        per_annotator.append(min(1.0, matches / 3.0))
    return sum(per_annotator) / len(per_annotator)


import math, types
from .io import require as need

def bounded_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                    position_embeddings=None):
    import torch
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

    need(not self.training and not torch.is_grad_enabled(),
         'bounded visual attention is inference-only')
    need(hidden_states.dtype == torch.float32, 'exact FP32 visual inputs required')
    length = hidden_states.shape[0]
    boundaries = cu_seqlens.tolist()
    need(boundaries and boundaries[0] == 0 and boundaries[-1] == length
         and all(type(x) is int for x in boundaries)
         and all(a < b for a, b in zip(boundaries, boundaries[1:])),
         'invalid original visual segment boundaries')
    q, k, v = self.qkv(hidden_states).reshape(
        length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        need(rotary_pos_emb is not None, 'original rotary embeddings absent')
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos, sin = emb.cos().float(), emb.sin().float()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        keys = k[start:end].transpose(0, 1).transpose(1, 2)
        values = v[start:end].transpose(0, 1)
        chunks = []
        for lo in range(start, end, self._revision_query_chunk):
            hi = min(end, lo + self._revision_query_chunk)
            queries = q[lo:hi].transpose(0, 1)
            weights = torch.matmul(queries, keys) / math.sqrt(self.head_dim)
            weights = torch.nn.functional.softmax(weights, dim=-1,
                                                   dtype=torch.float32).to(q.dtype)
            chunks.append(torch.matmul(weights, values).transpose(0, 1))
        segments.append(torch.cat(chunks, dim=0))
    attention = torch.cat(segments, dim=0).reshape(length, -1)
    return self.proj(attention)

def install_bounded_attention(model, query_chunk=256):
    import torch
    from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionAttention

    need(type(query_chunk) is int and 0 < query_chunk <= 256,
         'query chunk must be in [1,256]')
    need(not any(p.requires_grad for p in model.parameters())
         and not any(m.training for m in model.modules()),
         'frozen evaluation model required')
    modules = [(name, module) for name, module in model.named_modules()
               if type(module) is VisionAttention]
    need(modules, 'original Qwen2-VL eager visual attention absent')
    for name, module in modules:
        need(not hasattr(module, '_revision_query_chunk'),
             'visual attention already modified: ' + name)
        need(all(p.dtype == torch.float32 for p in module.parameters()),
             'visual parameter precision changed: ' + name)
    for _, module in modules:
        module._revision_query_chunk = query_chunk
        module.forward = types.MethodType(bounded_forward, module)
    return {'implementation': 'qwen2-vl-block-diagonal-query-chunks-v1',
            'query_chunk': query_chunk, 'modules': [name for name, _ in modules],
            'all_original_keys_per_query': True, 'pixels_or_tokens_changed': False,
            'language_attention_changed': False}

# Frozen input preparation is separate from model evaluation. No outcome is used.
def legacy_image_id(image):
    import hashlib
    h=hashlib.sha256();h.update(str(image.mode).encode('ascii'));h.update(str(image.size).encode('ascii'));h.update(image.tobytes())
    return h.hexdigest()


def image_path(data_root,identifier):
    import hashlib
    from pathlib import Path
    return Path(data_root)/'q11'/'images'/(hashlib.sha256(identifier.encode()).hexdigest()+'.png')


def image_manifest(task):
    from .io import read_json,source_path
    name='retention_images.json' if task in ('textvqa','scienceqa') else 'images.json.gz'
    return read_json(source_path('data/protocols/q11/'+name))


def open_image_loader(data_root,task):
    from .io import sha256
    from PIL import Image
    manifest=image_manifest(task)
    def load(identifier):
        path=image_path(data_root,identifier);record=manifest[identifier]
        need(path.is_file(),'Missing Q11 image; run open_tasks.prepare_data first: '+str(path))
        need(sha256(path)==record['sha256'],'Q11 source image bytes differ: '+identifier)
        with Image.open(path) as im:return im.convert('RGB')
    return load


def render_rule(scene,config):
    from PIL import Image,ImageDraw
    size=config['img_size'];image=Image.new('RGB',(size,size),tuple(scene['background']));draw=ImageDraw.Draw(image)
    for obj in scene['objects']:
        x,y=obj['center'];r=obj['radius']
        draw.ellipse((x-r,y-r,x+r,y+r),fill=tuple(config['color_rgb'][obj['color']]))
    return image


def prepare_data(data_root,task,*,cache_dir=None,parquet_root=None):
    """Materialize only images needed by the frozen schedules, verifying bytes.

    Dataset downloads use the recorded HF commits. VQAv2 image identities are
    located in the first two recorded validation parquet shards, preserving the
    original native image pixels. Rendered rule images use frozen scene geometry.
    """
    from pathlib import Path
    from .io import read_json,write_json,source_path,sha256
    manifest=image_manifest(task)
    name='retention_episodes.json' if task in ('textvqa','scienceqa') else 'episodes.json.gz'
    episodes=[e for e in read_json(source_path('data/protocols/q11/'+name)) if e['task']==task]
    need(episodes,'No frozen Q11 image schedule')
    wanted={i for e in episodes for i in e['images']}
    def save(identifier,image):
        path=image_path(data_root,identifier);path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():image.save(path,format='PNG')
        need(sha256(path)==manifest[identifier]['sha256'],'Reconstructed PNG differs: '+identifier+'; use recorded Pillow version')
    missing={i for i in wanted if not image_path(data_root,i).exists()}
    if task=='rule':
        for partition in ['development','confirmatory']:
            name='rule_scenes_'+partition+'.json'+('.gz' if partition=='confirmatory' else '')
            scenes=read_json(source_path('data/protocols/q11/'+name))
            for key,scene in scenes['scenes'].items():
                if key not in missing:continue
                image=render_rule(scene,scenes['config'])
                need(legacy_image_id(image)==scene['image_id'],'Frozen rule geometry changed')
                save(key,image);missing.remove(key)
    elif missing:
        from datasets import load_dataset
        if task=='vqav2':
            from huggingface_hub import hf_hub_download
            source=read_json(source_path('data/protocols/q11/vqav2_source.json'))
            files=[]
            for name,expected in source['source_transport']['local_parquet_sha256'].items():
                path=str(Path(parquet_root)/name) if parquet_root else hf_hub_download(repo_id=source['dataset'],repo_type='dataset',revision=source['dataset_revision'],filename='data/'+name,cache_dir=cache_dir)
                need(sha256(path)==expected,'Pinned VQAv2 parquet differs: '+name);files.append(path)
            import pyarrow.parquet as parquet
            from datasets import Image as DatasetImage
            decoder=DatasetImage()
            def native_images():
                for filename in files:
                    for batch in parquet.ParquetFile(filename).iter_batches(batch_size=64,columns=['image']):
                        for row in batch.to_pylist():yield decoder.decode_example(row['image'])
            for image in native_images():
                if image is None:continue
                identity=legacy_image_id(image)
                if identity in missing:save(identity,image);missing.remove(identity)
                if not missing:break
        else:
            source=read_json(source_path('data/protocols/q11/retention_provenance.json'))[task]
            targets={int(e['query_id']):e['images'][0] for e in episodes}
            stream=load_dataset(source['dataset'],revision=source['revision'],split=source['split'],streaming=True,cache_dir=cache_dir)
            for index,row in enumerate(stream):
                if index in targets and targets[index] in missing:
                    identifier=targets[index];image=row['image'].convert('RGB')
                    from .shapes import decoded_hash
                    need(decoded_hash(image)==manifest[identifier]['decoded_sha256'],'Pinned retention pixels differ')
                    save(identifier,image);missing.remove(identifier)
                if not missing:break
    need(not missing,'Pinned input sources did not contain all required images: '+str(len(missing)))
    for identifier in wanted:
        need(sha256(image_path(data_root,identifier))==manifest[identifier]['sha256'],'Existing cached image changed')
    result={'task':task,'images':len(wanted),'verified_png_sha256':True,'selection':'bundled frozen episode IDs'}
    write_json(Path(data_root)/'q11'/(task+'_materialization.json'),result)
    return result
