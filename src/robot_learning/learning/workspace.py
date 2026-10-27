"""Disk-backed training buffers and training-only incremental normalization."""
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from robot_learning.data.storage import read_json, sha256
from .dataset import load_episode
from .model import MLPPolicy, INPUT_NAMES, OUTPUT_NAMES


def mapped_training(directory, records, output, preprocessing_from=None, window=None):
    workspace = Path(output)/'work'
    workspace.mkdir()
    rows = sum(r['rows'] for r in records)
    x = np.lib.format.open_memmap(workspace/'x_raw.npy', mode='w+', dtype='float64', shape=(rows,21))
    y = np.lib.format.open_memmap(workspace/'y_raw.npy', mode='w+', dtype='float64', shape=(rows,9))
    xs = StandardScaler(); ys = StandardScaler(); offset = 0
    for record in records:
        a,b,_ = load_episode(directory,record,window=window); end = offset+len(a)
        x[offset:end] = a; y[offset:end] = b
        if preprocessing_from is None:
            xs.partial_fit(a); ys.partial_fit(b)
        offset = end
    if offset != rows:
        raise ValueError('Mapped row count differs from inventory')
    x.flush(); y.flush()
    note = {'mode':'disk_mapped','raw_dtype':'float64','normalized_dtype':'float32',
            'normalization':'StandardScaler.partial_fit on complete training episodes only',
            'temporary_workspace':'work/; reconstructible from immutable dataset; removed after success'}
    if preprocessing_from is not None:
        source = Path(preprocessing_from).resolve(); old = read_json(source/'run.json')
        oldsplit = read_json(source/'split.json')
        if (old['status']!='complete' or sha256(source/'model.npz')!=old['model_sha256']
                or sha256(source/'split.json')!=old['split_sha256']):
            raise ValueError('Changed preprocessing reference')
        if [r['npz_sha256'] for r in records] != [r['npz_sha256'] for r in oldsplit['training']]:
            raise ValueError('Frozen normalizers require identical ordered training trajectories')
        model = MLPPolicy.load(source/'model.npz'); info = read_json(source/'model_info.json')
        for scaler,mean,scale,names,constant_key in [
            (xs,model.x_mean,model.x_scale,INPUT_NAMES,'constant_training_inputs'),
            (ys,model.y_mean,model.y_scale,OUTPUT_NAMES,'constant_training_outputs')]:
            scaler.mean_=mean.copy(); scaler.scale_=scale.copy(); scaler.var_=scale**2
            for name in info[constant_key]: scaler.var_[names.index(name)]=0.
            scaler.n_features_in_=len(mean); scaler.n_samples_seen_=rows
        note.update(normalization='Exact frozen training-only normalizers; identical training inventory',
                    preprocessing_run=str(source),preprocessing_model_sha256=old['model_sha256'])
    return x,y,xs,ys,note


def normalized_maps(x,y,x_scaler,y_scaler,output,chunk_rows=65536):
    paths = [Path(output)/'work/x_normalized.npy',Path(output)/'work/y_normalized.npy']
    arrays=[]
    for source,scaler,path in zip([x,y],[x_scaler,y_scaler],paths):
        result=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=source.shape)
        for start in range(0,len(source),chunk_rows):
            end=min(start+chunk_rows,len(source))
            result[start:end]=scaler.transform(source[start:end]).astype(np.float32)
        result.flush();arrays.append(result)
    return tuple(arrays)
