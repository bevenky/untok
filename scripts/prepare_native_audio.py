#!/usr/bin/env python3
"""Prepare bounded real audio for native-tokenizer engineering checks.

FLEURS test audio covers available languages; separate locale-proxy flags stop
reused recordings from masquerading as regional coverage. IndicVoices supplies
missing Indic scripts from clean, complete source transcripts. Train clips are
kept in a separate manifest and must not be used as accuracy evaluation data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import struct
import urllib.parse
import urllib.request

import prepare_fleurs_smoke as fleurs

IV_DATASET = "ai4bharat/IndicVoices"
IV_REVISION = "c96f9088f138cf89d419da7e8e643e1f05c00a87"
MODEL_REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"
# This is the published support list, not every key in the prompt dictionary.
TIERS = {
    "transcription_ready": "en-US en-GB es-US es-ES fr-FR fr-CA it-IT pt-BR pt-PT nl-NL de-DE tr-TR ru-RU ar-AR hi-IN ja-JP ko-KR vi-VN uk-UA".split(),
    "broad_coverage": "pl-PL sv-SE cs-CZ nb-NO da-DK bg-BG fi-FI hr-HR sk-SK zh-CN hu-HU ro-RO et-EE".split(),
    "adaptation_ready": "el-GR lt-LT lv-LV mt-MT sl-SI he-IL th-TH nn-NO".split(),
}
LOCALE_CONFIG = {
    "en-US":"en_us", "en-GB":"en_us", "es-US":"es_419", "es-ES":"es_419",
    "fr-FR":"fr_fr", "fr-CA":"fr_fr", "it-IT":"it_it", "pt-BR":"pt_br", "pt-PT":"pt_br",
    "nl-NL":"nl_nl", "de-DE":"de_de", "tr-TR":"tr_tr", "ru-RU":"ru_ru", "ar-AR":"ar_eg",
    "hi-IN":"hi_in", "ja-JP":"ja_jp", "ko-KR":"ko_kr", "vi-VN":"vi_vn", "uk-UA":"uk_ua",
    "pl-PL":"pl_pl", "sv-SE":"sv_se", "cs-CZ":"cs_cz", "nb-NO":"nb_no", "da-DK":"da_dk",
    "bg-BG":"bg_bg", "fi-FI":"fi_fi", "hr-HR":"hr_hr", "sk-SK":"sk_sk", "zh-CN":"cmn_hans_cn",
    "hu-HU":"hu_hu", "ro-RO":"ro_ro", "et-EE":"et_ee", "el-GR":"el_gr", "lt-LT":"lt_lt",
    "lv-LV":"lv_lv", "mt-MT":"mt_mt", "sl-SI":"sl_si", "he-IL":"he_il", "th-TH":"th_th", "nn-NO":"nb_no",
}
PROXIES = {
    "en-GB":"FLEURS publishes US English, not a separate UK recording set.",
    "es-US":"FLEURS publishes Latin American Spanish, not a US-specific recording set.",
    "es-ES":"FLEURS publishes Latin American Spanish, not a Spain-specific recording set.",
    "fr-CA":"FLEURS publishes France French, not a Canadian recording set.",
    "pt-PT":"FLEURS publishes Brazilian Portuguese, not a Portugal recording set.",
    "ar-AR":"Generic model Arabic prompt tested with the FLEURS ar_eg source; regional breadth is not established.",
    "nn-NO":"FLEURS publishes Norwegian Bokmal, not Nynorsk. This is a prompt-path check only.",
}
INDIC_FLEURS = {lang:config for lang,config in zip(
    "as bn gu hi kn ml mr ne or pa ta te ur".split(),
    "as_in bn_in gu_in hi_in kn_in ml_in mr_in ne_np or_in pa_in ta_in te_in ur_pk".split())}
IV_CONFIG = dict(zip("brx doi kok ks mai mni sa sat sd".split(),
                     "bodo dogri konkani kashmiri maithili manipuri sanskrit santali sindhi".split()))
LOCALE = {lang:lang+"-IN" for lang in set(INDIC_FLEURS)|set(IV_CONFIG)}
LOCALE.update({"ne":"ne-NP", "ur":"ur-PK"})


def sha(data): return hashlib.sha256(data).hexdigest()
def write(path, obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
def write_rows(path, rows): path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))


def get_json(url, token=None):
    headers={"User-Agent":"untok-native-audio/1"}
    if token: headers["Authorization"]="Bearer "+token
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as stream:
        return json.load(stream)


def audio_info(data):
    if data.startswith(b"RIFF"):
        return fleurs.wav_info(data)
    if not data.startswith(b"fLaC") or len(data)<42 or data[4]&127:
        raise ValueError("Expected source WAV or FLAC with STREAMINFO")
    size=int.from_bytes(data[5:8],"big")
    if size!=34: raise ValueError("Unexpected FLAC STREAMINFO")
    info=int.from_bytes(data[18:26],"big")
    rate=(info>>44)&0xfffff; channels=((info>>41)&7)+1
    bits=((info>>36)&31)+1; samples=info&((1<<36)-1)
    if rate!=16000 or channels!=1 or not samples:
        raise ValueError("Expected mono16kHz source audio")
    return {"sample_rate":rate,"channels":channels,"encoding":"FLAC","bits_per_sample":bits,
            "num_samples":samples,"duration":samples/rate}


def script_present(text, target):
    return any(any(lo<=ord(c)<=hi for lo,hi in target['ranges']) and c.isalpha() for c in text)


def native_labels(row, adapter):
    ids=adapter.text_to_ids(row['text'])
    return dict(row,native_token_ids=ids,new_native_token_ids=sorted({i for i in ids if i>=adapter.id_map.hf_pad_id}),
                contains_unknown_token=adapter.unk_id in ids,tokenizer_sha256=adapter.id_map.tokenizer_sha256,
                text_is_complete_source_transcript=True,tokenizer_annotation_cleanup_applied=False)


def reuse_clips(paths, config, split, count, args):
    clips=[];seen=set()
    for manifest in paths:
        if not manifest.exists(): continue
        for line in manifest.open():
            row=json.loads(line)
            if row.get('source_dataset')!=fleurs.DATASET or row.get('source_revision')!=fleurs.REVISION or row.get('source_split')!=split:
                continue
            if f'/data/{config}/' not in row.get('source_tsv_url',''): continue
            if not args.min_duration<=row['duration']<=args.max_duration: continue
            if row['source_audio_filename'] in seen: continue
            old=Path(row['audio_filepath'])
            if not old.exists(): old=manifest.parent/row['audio']
            if not old.exists() or sha(old.read_bytes())!=row['audio_sha256']: continue
            relative=Path('audio')/config/split/row['source_audio_filename']; new=args.output/relative
            new.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,new)
            item={k:v for k,v in row.items() if k not in ['canonical_token_ids','new_canonical_token_ids','contains_unknown_token']}
            item.update(audio=str(relative),audio_filepath=str(new.resolve()),reused_from_manifest=str(manifest.resolve()))
            clips.append(item);seen.add(row['source_audio_filename'])
            if len(clips)==count:return clips
    return []


def prepare_fleurs(config, split, count, args, registry):
    old=reuse_clips(args.reuse_jsonl,config,split,count,args)
    if old:return old,{'config':config,'split':split,'selected':len(old),'reused_verified_clips':True}
    return fleurs.prepare_split(config,split,count,args,None,registry)


def prepare_iv(lang, split, count, args, target, token, excluded_speakers):
    if not token:raise PermissionError("IndicVoices requires authorized Hugging Face access")
    config=IV_CONFIG[lang];selected=[];speakers=set(excluded_speakers);texts=set();scanned=0
    skip=Counter();urls=[]
    for offset in range(0,args.max_iv_rows,100):
        query=urllib.parse.urlencode({'dataset':IV_DATASET,'config':config,'split':split,'offset':offset,'length':100})
        url='https://datasets-server.huggingface.co/rows?'+query
        page=get_json(url,token);urls.append(url)
        for item in page['rows']:
            scanned+=1;row=item['row'];text=row.get('unsanitized_verbatim')
            if item.get('truncated_cells'):skip['truncated_source_fields']+=1;continue
            if not isinstance(text,str) or not text.strip() or any(c in text for c in '<>[]{}'):
                skip['empty_or_annotation_markup']+=1;continue
            if row.get('lang')!=lang or not script_present(text,target):skip['wrong_language_or_script']+=1;continue
            if not args.min_duration<=row['duration']<=args.max_duration:skip['duration']+=1;continue
            speaker=row.get('speaker_id')
            if not speaker or speaker in speakers or text in texts:skip['repeated_speaker_or_text']+=1;continue
            assets=row.get('audio_filepath')
            if not isinstance(assets,list) or len(assets)!=1:raise ValueError('Unexpected viewer audio field')
            audio_url=assets[0]['src'];parsed=urllib.parse.urlsplit(audio_url)
            marker=f'/--/{IV_REVISION}/--/{config}/{split}/{item["row_idx"]}/'
            if parsed.scheme!='https' or parsed.hostname!='datasets-server.huggingface.co' or marker not in parsed.path:
                raise ValueError('Viewer audio is not bound to the expected pinned revision and row')
            with urllib.request.urlopen(urllib.request.Request(audio_url,headers={'Authorization':'Bearer '+token}),timeout=60) as response:
                data=response.read(10*1024*1024+1)
            if len(data)>10*1024*1024:raise ValueError('Individual audio byte cap exceeded')
            info=audio_info(data)
            if row.get('samples') is not None and info['num_samples']!=row['samples']:
                raise ValueError('Source audio samples disagree with metadata')
            if abs(info['duration']-row['duration'])>.02:raise ValueError('Source audio duration mismatch')
            suffix='.flac' if data.startswith(b'fLaC') else '.wav'
            filename=f'{item["row_idx"]}{suffix}';relative=Path('audio')/'indicvoices'/config/split/filename
            path=args.output/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            preserved={k:v for k,v in row.items() if k!='audio_filepath'}
            result={'id':f'indicvoices-{config}-{split}-{item["row_idx"]}','audio':str(relative),
                    'audio_filepath':str(path.resolve()),'audio_sha256':sha(data),'target_lang':LOCALE[lang],
                    'language':lang,'language_name':target['name'],'script':target['script'],
                    'text':text,'raw_transcription':text,'publisher_verbatim':row.get('verbatim'),
                    'publisher_normalized_transcription':row.get('normalized'),'source_text_field':'unsanitized_verbatim',
                    'speaker_id':speaker,'scenario':row.get('scenario'),'district':row.get('district'),'state':row.get('state'),
                    'raw_source_metadata':preserved,'source_dataset':IV_DATASET,'source_revision':IV_REVISION,
                    'source_split':split,'source_row_index':item['row_idx'],'source_viewer_url':url,
                    'source_audio_url_without_signature':urllib.parse.urlunsplit((parsed.scheme,parsed.netloc,parsed.path,'','')),
                    'source_metadata_sha256':sha(json.dumps(preserved,ensure_ascii=False,sort_keys=True).encode()),
                    'source_asset_note':'Individual pinned-revision dataset-viewer asset; source full Parquet shard was not downloaded.',
                    'license':'CC-BY-4.0','purpose':'training_smoke' if split=='train' else 'migration_development_check',**info}
            selected.append(result);speakers.add(speaker);texts.add(text)
            if len(selected)==count:return selected,{'language':lang,'config':config,'split':split,'selected':len(selected),'rows_scanned':scanned,'skipped':dict(skip),'metadata_pages':urls}
        if offset+100>=page.get('num_rows_total',0):break
    raise ValueError(f'Insufficient clean complete distinct-speaker clips for {lang}/{split}: {len(selected)}/{count}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-processor',type=Path,required=True)
    p.add_argument('--tokenizer-bundle',type=Path,required=True)
    p.add_argument('--targets',type=Path,default=Path('configs/build.json'))
    p.add_argument('--reuse-jsonl',type=Path,nargs='*',default=[])
    p.add_argument('--hf-token-file',type=Path)
    p.add_argument('--eval-per-profile',type=int,default=2)
    p.add_argument('--train-per-indic',type=int,default=1)
    p.add_argument('--min-duration',type=float,default=3.)
    p.add_argument('--max-duration',type=float,default=10.)
    p.add_argument('--max-download-mb-per-archive',type=int,default=64)
    p.add_argument('--max-iv-rows',type=int,default=1000)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if min(args.eval_per_profile,args.train_per_indic)<1 or args.min_duration<=0 or args.max_duration<args.min_duration or args.workers<1:
        p.error('Request positive counts, durations and workers')
    if args.output.exists() and any(args.output.iterdir()):p.error('Use an empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    from untok.unigram import NativeTokenizerAdapter
    adapter=NativeTokenizerAdapter(args.tokenizer_bundle)
    registry=json.loads(args.source_processor.read_text())['prompt_dictionary']
    targets={r['language']:r for r in json.loads(args.targets.read_text())['targets']}
    assert len(LOCALE_CONFIG)==40 and set(LOCALE_CONFIG)=={l for v in TIERS.values() for l in v}
    assert set(LOCALE_CONFIG)<=set(registry)
    assert set(targets)==set(INDIC_FLEURS)|set(IV_CONFIG)
    token=args.hf_token_file.read_text().strip() if args.hf_token_file else None
    config_info={}
    for locale,config in LOCALE_CONFIG.items():
        config_info.setdefault(config,(locale.split('-')[0],locale, 'unspecified',locale))
    for lang,config in INDIC_FLEURS.items():config_info[config]=(lang,targets[lang]['name'],targets[lang]['script'],LOCALE[lang])
    fleurs.PROFILES=config_info
    inventory={'source_model_revision':MODEL_REVISION,'source_model_card':f'https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/{MODEL_REVISION}/README.md',
               'original_locales':[{'locale':loc,'tier':tier,'fleurs_config':LOCALE_CONFIG[loc],'locale_audio_match':'proxy' if loc in PROXIES else 'matching_published_locale','limitation':PROXIES.get(loc),'prompt_id':registry[loc]} for tier,locs in TIERS.items() for loc in locs],
               'indic_profiles':[{'language':lang,'target_lang':LOCALE[lang],'script':targets[lang]['script'],'source':fleurs.DATASET if lang in INDIC_FLEURS else IV_DATASET,'config':INDIC_FLEURS.get(lang,IV_CONFIG.get(lang)),'source_prompt_available':LOCALE[lang] in registry} for lang in targets],
               'sindhi_note':'FLEURS sd_in is Arabic script; it is not used for the Devanagari Sindhi profile.'}
    write(args.output/'inventory.json',inventory)
    clips={};provenance=[];errors=[]
    jobs=[(config,'test',args.eval_per_profile) for config in sorted(config_info)]
    jobs += [(config,'train',args.train_per_indic) for config in sorted(set(INDIC_FLEURS.values()))]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(prepare_fleurs,c,s,n,args,registry):(c,s) for c,s,n in jobs}
        for future in as_completed(futures):
            config,split=futures[future]
            try:
                selected,source=future.result();clips[config,split]=selected;provenance.append(source)
                print(json.dumps({'dataset':fleurs.DATASET,'config':config,'split':split,'clips':len(selected)}),flush=True)
            except Exception as error:
                errors.append({'dataset':fleurs.DATASET,'config':config,'split':split,'exception':type(error).__name__,'http_status':getattr(error,'code',None)})
                print(json.dumps(errors[-1]),flush=True)
    iv={}
    for lang in IV_CONFIG:
        excluded=set()
        for split,count in [('valid',args.eval_per_profile),('train',args.train_per_indic)]:
            try:
                selected,source=prepare_iv(lang,split,count,args,targets[lang],token,excluded)
                iv[lang,split]=selected;provenance.append(source)
                if split=='valid':excluded={r['speaker_id'] for r in selected}
                print(json.dumps({'dataset':IV_DATASET,'language':lang,'split':split,'clips':len(selected)}),flush=True)
            except Exception as error:
                errors.append({'dataset':IV_DATASET,'language':lang,'split':split,'exception':type(error).__name__,'http_status':getattr(error,'code',None)})
                print(json.dumps(errors[-1]),flush=True)
    original=[]
    for tier,locales in TIERS.items():
        for locale in locales:
            for row in clips.get((LOCALE_CONFIG[locale],'test'),[]):
                original.append(native_labels(dict(row,id=row['id']+'-prompt-'+locale,target_lang=locale,cohort='original_locale',support_tier=tier,
                    locale_audio_match='proxy' if locale in PROXIES else 'matching_published_locale',locale_coverage_limitation=PROXIES.get(locale),
                    source_locale_matches_target=locale not in PROXIES,locale_specific_accuracy_evidence=False),adapter))
    indicative=[];training=[]
    for lang in targets:
        source=clips if lang in INDIC_FLEURS else iv; key=INDIC_FLEURS.get(lang,lang)
        split='test' if lang in INDIC_FLEURS else 'valid'
        for row in source.get((key,split),[]):indicative.append(native_labels(dict(row,target_lang=LOCALE[lang],language=lang,cohort='indic_profile',source_prompt_available=LOCALE[lang] in registry),adapter))
        for row in source.get((key,'train'),[]):training.append(native_labels(dict(row,target_lang=LOCALE[lang],language=lang,cohort='indic_training_smoke',source_prompt_available=LOCALE[lang] in registry),adapter))
    write_rows(args.output/'original-locale-eval.jsonl',original)
    write_rows(args.output/'indic-eval.jsonl',indicative)
    write_rows(args.output/'training-smoke.jsonl',training)
    write_rows(args.output/'physical-clips.jsonl',[r for rs in list(clips.values())+list(iv.values()) for r in rs])
    missing={'original_prompt_profiles':[l for l in LOCALE_CONFIG if sum(r['target_lang']==l for r in original)!=args.eval_per_profile],
             'indic_eval_profiles':[l for l in targets if sum(r['language']==l for r in indicative)!=args.eval_per_profile],
             'indic_train_profiles':[l for l in targets if sum(r['language']==l for r in training)!=args.train_per_indic],
             'original_exact_locale_coverage':PROXIES}
    report={'status':'complete_engineering_manifest' if not errors and not any(missing[k] for k in ['original_prompt_profiles','indic_eval_profiles','indic_train_profiles']) else 'incomplete',
            'selection_policy':'First eligible source/archive rows by duration, nonempty complete text and expected script; no acoustic predictions used. IV requires distinct source speakers and excludes valid speakers from train selection.',
            'evaluation_is_engineering_smoke_not_release_accuracy':True,'original_locale_rows':len(original),'indic_eval_rows':len(indicative),'indic_train_rows':len(training),
            'distinct_audio_files':len({r['audio_sha256'] for r in original+indicative+training}),'total_distinct_audio_seconds':sum({r['audio_sha256']:r['duration'] for r in original+indicative+training}.values()),
            'missing':missing,'errors':errors,'sources':provenance,'tokenizer_sha256':adapter.id_map.tokenizer_sha256,
            'source_processor_sha256':sha(args.source_processor.read_bytes()),'targets_sha256':sha(args.targets.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),
            'fleurs_helper_sha256':sha(Path(fleurs.__file__).read_bytes()),'fleurs_revision':fleurs.REVISION,'indicvoices_revision':IV_REVISION,
            'source_transcripts_modified':False,'signed_asset_queries_saved':False,'tokenizer_cleaned_partial_transcripts_used':False,
            'training_and_evaluation_manifests_separate':True,'training_performed':False,'predictions_generated':False,
            'file_sha256':{p.name:sha(p.read_bytes()) for p in args.output.iterdir() if p.is_file()}}
    write(args.output/'coverage.json',report)
    print(json.dumps({k:report[k] for k in ['status','original_locale_rows','indic_eval_rows','indic_train_rows','distinct_audio_files','total_distinct_audio_seconds']}))
    return 0 if report['status']=='complete_engineering_manifest' else 2


if __name__=='__main__':raise SystemExit(main())
