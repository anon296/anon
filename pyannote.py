from pyannote.audio import Pipeline
import torch
import numpy as np
import pandas as pd
import os
import json


from pyannote.core.segment import Segment
manlist = [(Segment(0.857844, 1.88722), 'A', 'SPEAKER_01'), (Segment(2.71409, 4.36784), 'B', 'SPEAKER_01'), (Segment(53.3391, 55.3135), 'C', 'SPEAKER_03'), (Segment(55.5835, 56.1066), 'D', 'SPEAKER_03'), (Segment(56.5453, 59.971), 'E', 'SPEAKER_03'), (Segment(61.4391, 62.671), 'F', 'SPEAKER_04'), (Segment(64.5272, 65.776), 'G', 'SPEAKER_04'), (Segment(65.9785, 68.3747), 'H', 'SPEAKER_04'), (Segment(69.1003, 69.3535), 'I', 'SPEAKER_04'), (Segment(70.6528, 73.6735), 'J', 'SPEAKER_04'), (Segment(74.2303, 74.4497), 'K', 'SPEAKER_03'), (Segment(77.5885, 77.6897), 'L', 'SPEAKER_01'), (Segment(78.2128, 78.4491), 'M', 'SPEAKER_03'), (Segment(80.6428, 81.9085), 'N', 'SPEAKER_02'), (Segment(83.3766, 83.7478), 'O', 'SPEAKER_01'), (Segment(86.4141, 87.4941), 'P', 'SPEAKER_02'), (Segment(92.0166, 97.2985), 'Q', 'SPEAKER_00'), (Segment(98.2435, 99.9647), 'R', 'SPEAKER_00')]

epname = 'oltl-10-18-10'
if os.path.exists(df_path:=f'{epname}-speakerturns.csv'):
    df = pd.read_csv(df_path, index_col=0)
else:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
    pipeline.to(torch.device("cuda"))
    diarization = pipeline(f"SummScreen/audio/{epname}.wav")

    prev_speaker=None
    speaker_starttime = np.inf
    speaker_endtime = 0
    turns_list = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_starttime = min(speaker_starttime, turn.start)
        speaker_endtime = max(speaker_endtime, turn.end)
        if speaker != prev_speaker:
            prev_speaker = speaker
            if prev_speaker is not None:
                print(f"start={speaker_starttime:.2f}s stop={speaker_endtime:.2f}s speaker_{speaker}")
                turns_list.append({'start':speaker_starttime, 'end': speaker_endtime, 'sid':speaker.removeprefix('SPEAKER_')})
                speaker_starttime = np.inf
                speaker_endtime = 0
    df = pd.DataFrame(turns_list)
    df.to_csv(df_path)

breakpoint()
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...
