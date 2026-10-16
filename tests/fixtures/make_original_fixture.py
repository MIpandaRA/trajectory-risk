"""How tests/fixtures/original_wrappers_fixture.npz was made.

Run from inside the ORIGINAL prototype folder (live_tracking_system/, tag
v0-vibecoded), not this repo:

    cd live_tracking_system && python make_original_fixture.py original_wrappers_fixture.npz

It calls the prototype's own inference.py / social_inference.py with a fixed
seed on a fixed input and saves their intermediate tensors and raw samples.
"""
import sys, numpy as np, torch, math
sys.path.insert(0, '.')
from inference import LiveContextVAE, undo_rotation
from social_inference import LiveSocialVAE
cv = LiveContextVAE('models_contextvae/config/nuscenes_eval_3s_k10.py','models_contextvae/checkpoints/ckpt-best','maps/T11/semantic_map.pkl')
sv = LiveSocialVAE('models_socialvae/config/sdd.py','models_socialvae/checkpoints/ckpt-best')
d = np.array([-4.5,-7.6]); d/=np.linalg.norm(d)
start = np.array([73.6,53.3])
ego = [tuple(start + d*1.25*i) for i in range(cv.buffer_len)]
heading = math.atan2(d[1], d[0])
nb1 = [tuple(start + np.array([-3.0, 1.0]) + d*1.0*i) for i in range(cv.buffer_len)]
nb2 = [None, None] + [tuple(np.array([60.0, 40.0]) + np.array([0.8, 0.2])*i) for i in range(cv.buffer_len-2)]
out = {}
pos = np.asarray(ego); ext = cv._extend(pos); hist = ext[-cv.ob_horizon:]; pivot = hist[0,:2].copy(); angle = -heading
out['cv_hist_rot'] = cv._rotate(hist, pivot, angle)
out['cv_map_crop'] = cv._map_crop(pivot[0], pivot[1], angle).numpy()
torch.manual_seed(0)
# raw samples via the model path, replicating predict() up to the model call
import inference
orig_cluster = inference.clustering
captured = {}
def cap(y, k):
    captured['raw'] = y.copy(); return orig_cluster(y, k)
inference.clustering = cap
torch.manual_seed(0); cands = cv.predict(ego, heading, [nb1, nb2])
out['cv_raw_world'] = captured['raw']
ped = [tuple(np.array([20.0, 15.0]) + np.array([1.3*0.4, 0.0])*i) for i in range(sv.buffer_len)]
pnb = [[tuple(np.array([25.0, 18.0]) - np.array([0.5,0.1])*i) for i in range(sv.buffer_len)]]
import social_inference
cap2 = {}
orig_fpc = social_inference.LiveSocialVAE._confidence_weighted_fpc
def cap_fpc(y, k):
    cap2['raw'] = y.copy(); return orig_fpc(y, k)
social_inference.LiveSocialVAE._confidence_weighted_fpc = staticmethod(cap_fpc)
torch.manual_seed(0); pc = sv.predict(ped, pnb)
out['sv_raw'] = cap2['raw']
np.savez_compressed(sys.argv[1], ego=np.array(ego), heading=heading, nb1=np.array(nb1),
    nb2=np.array([p if p is not None else (np.nan,np.nan) for p in nb2]), ped=np.array(ped), ped_nb=np.array(pnb[0]), **out)
print({k: v.shape for k, v in out.items()})
print('cv top', cands[0].confidence, cands[0].trajectory[-1], 'sv top', pc[0].confidence)
