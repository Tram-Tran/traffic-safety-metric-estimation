import os
import numpy as np
import motmetrics as mm
from motmetrics import distances

def prepare_accumulator(truths, hyp, meta=None):
    ignored_regions = meta['ignored_regions'] if meta != None else None

    acc = mm.MOTAccumulator(auto_id=True)

    for i in range(len(truths)):
        hyp_ids = []
        hyp_bboxes = []
        for frame in hyp[i]['frames']:
            hyp_id, hyp_bbox = frame['id'], frame['bbox']
            [x, y, w, h]= hyp_bbox
            valid = True
            if ignored_regions != None:
                for region in ignored_regions:
                    [ix, iy, iw, ih] = region
                    if x >= ix and y >= iy and (x + w) <= (ix + iw) and (y + h) <= (iy + ih):
                        valid = False
            if valid:
                hyp_ids.append(hyp_id)
                hyp_bboxes.append(hyp_bbox)

        truth_ids = [frame['id'] for frame in truths[i] if frame['class'] in ['car', 'truck', 'bus']]

        dw, dh = len(truth_ids), len(hyp_ids)
        distance_array = np.zeros(shape=(dw, dh))
        truth_bboxes = [frame['bbox'] for frame in truths[i] if frame['class'] in ['car', 'truck', 'bus']]
        distance_array = distances.iou_matrix(truth_bboxes, hyp_bboxes)

        acc.update(
            truth_ids,                  # Ground truth objects in this frame
            hyp_ids,                    # Detector hypotheses in this frame
            distance_array
        )

    return acc

if __name__ == '__main__':
    directory = 'datasets'
    label_path = 'DETRAC-Test-npy'
    tracklet_path = 'UAtestSORTbb/tracklets'

    print(f'Evaluating {tracklet_path}')

    list_tracklets = os.listdir(f'{directory}/{tracklet_path}')
    accs = []
    names = []
    for tracks in list_tracklets:
        if tracks.endswith('.npy'):
            sequence_name = tracks[:9]

            hyp = np.load(f'{directory}/{tracklet_path}/{tracks}', allow_pickle=True)
            truths = np.load(f'{directory}/{label_path}/{sequence_name}.npy', allow_pickle=True)
            meta = np.load(f'{directory}/{label_path}/{sequence_name}-metadata.npy', allow_pickle=True).item()

            accs.append(prepare_accumulator(truths, hyp, meta))
            names.append(sequence_name)

    mh = mm.metrics.create()
    # MOTP is the iou distance, which is 1 - iou
    summary = mh.compute_many(accs, metrics=['num_frames', 'mota', 'motp', 'precision', 'recall'], names=names)

    print(summary.describe())
    print('Weighted average MOTP over num frames:', (summary['num_frames'] * summary['motp']).sum() / summary['num_frames'].sum())
    print('Weighted average precision over num frames:', (summary['num_frames'] * summary['precision']).sum() / summary['num_frames'].sum())
    summary.to_csv(f'{directory}/{tracklet_path}/motmetrics.csv', sep='\t', encoding='utf-8', header=True)