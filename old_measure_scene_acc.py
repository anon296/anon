import json
from nltk.tokenize import sent_tokenize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import pandas as pd
import numpy as np
#from utils import get_all_testnames
from datasets import load_dataset
from difflib import SequenceMatcher
from align_vid_and_transcripts import align
import re
from tqdm import tqdm
import argparse
from dl_utils.label_funcs import accuracy


def get_all_testnames():
    with open('moviesumm_testset_names.txt') as f:
        official_names = f.read().split('\n')
    with open('clean-vid-names-to-command-line-names.json') as f:
        clean2cl = json.load(f)
    #assert all([x in [y.split('_')[0] for y in official_names] for x in clean2cl.keys()])
    assert all(x in official_names for x in clean2cl.keys())
    test_vidnames = list(clean2cl.values())
    return test_vidnames, clean2cl

parser = argparse.ArgumentParser()
parser.add_argument('--only-fix-transcripts', action='store_true')
parser.add_argument('--ndps', type=int, default=99999)
ARGS = parser.parse_args()

test_vidnames, clean2cl = get_all_testnames()
cl2clean = {v:k for k,v in clean2cl.items()}

ds = load_dataset("rohitsaxena/MovieSum")

metric_names = ['acc', 'ari', 'nmi']
method_names = ['ours', 'uniform', 'uniform-oracle']
results = {mt: {mc:[] for mc in metric_names} for mt in method_names}

all_accs = []
all_nmis = []
all_aris = []
test_vidnames = ['the-sixth-sense_1999'] + test_vidnames

for vn in (pbar:=tqdm(test_vidnames[:ARGS.ndps])):
    #if vn!='somethings-gotta-give_2003':
        #continue
    with open(f'data/transcripts/{vn}-no-names.json') as f:
        transcript = json.load(f)['Transcript']

    scene_idx = 0
    scene_labels = []
    transcript = [line for line in transcript if line.strip()!='']
    transcript = sum([['[SCENE_BREAK]'] if line=='[SCENE_BREAK]' else [line.split(':')[0] + ': ' + x for x in sent_tokenize(line.split(':')[1])] for line in transcript], [])
    for t in transcript:
        if t=='[SCENE_BREAK]':
            scene_idx+=1
        else:
            scene_labels.append(scene_idx)

    gt_match_name = cl2clean[vn]
    gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
    gt_script = gt_match['script']
    gt_transcript = []
    speaker_name = ''
    gt_scene_idx = 0
    gt_scene_labels = []
    for l in gt_script.split('\n'):
        l = l.strip()
        if l.startswith('<scene>'):
            gt_scene_idx += 1
            #print(f'incrementing sidx to {gt_scene_idx} at line', l)
        elif l.startswith('<character>'):
            speaker_name = l.removeprefix('<character>').removesuffix('</character>')
        elif l.startswith('<dialogue>'):
            spoken = l.removeprefix('<dialogue>').removesuffix('</dialogue>')
            for sent in sent_tokenize(spoken):
                gt_transcript.append(f'{speaker_name}: {sent}')
                gt_scene_labels.append(gt_scene_idx)
                assert len(gt_transcript) == len(gt_scene_labels)
                #print(f'add dialogue line at {gt_scene_idx}: {sent}')

    transcript_no_sblines = [x for x in transcript if x != '[SCENE_BREAK]']
    alignment = align(transcript_no_sblines, gt_transcript)
    gt = np.array(gt_scene_labels)[alignment.index2]
    #for i,j in zip(alignment.index1, alignment.index2):
        #print(transcript_no_sblines[i], '&&&', gt_transcript[j])
    ours = np.array(scene_labels)[alignment.index1]
    ours2 = np.array(scene_labels)[alignment.index1]
    unif_baseline = np.linspace(0,30,len(gt)).astype(int)
    unifo_baseline = np.linspace(0, gt_scene_idx, len(gt)).astype(int)
    for mt, mtpred, in zip(method_names, [ours, unif_baseline, unifo_baseline]):
        #results[mt]['acc'].append(max(accuracy(gt, mtpred), accuracy(mtpred, gt)))
        results[mt]['acc'].append(accuracy(mtpred, gt))
        results[mt]['ari'].append(adjusted_rand_score(mtpred, gt))
        results[mt]['nmi'].append(normalized_mutual_info_score(mtpred, gt))

    #pbar.set_description(f'Acc: {np.array(results["ours"]["acc"]).mean():.3f}  NMI: {np.array(results["ours"]["nmi"]).mean():.3f}  ARI: {np.array(results["ours"]["ari"]).mean():.3f}')
    pbar.set_description('  '.join(f'{k}: ' +
        ' '.join(f'{k1}: {np.array(v1).mean():.3f}' for k1,v1 in v.items())
        for k,v in results.items()))
final_results = {k1:{k2:np.array(v2).mean() for k2,v2 in v1.items()} for k1,v1 in results.items()}
final_results = pd.DataFrame(final_results)
print(final_results)
final_results.to_csv('scenes-results.csv')

with open('names-results.json', 'w') as f:
    json.dump(results, f)
