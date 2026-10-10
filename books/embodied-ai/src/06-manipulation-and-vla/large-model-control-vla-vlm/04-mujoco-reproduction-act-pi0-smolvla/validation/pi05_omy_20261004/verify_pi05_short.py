"""Verify real weight updates and checkpoint inference after five OMY updates."""

import argparse
import json
import sys
import re
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from lerobot.datasets.lerobot_dataset import LeRobotDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pi05_omy_utils import load_policy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root', type=Path, required=True)
root = parser.parse_args().run_root.resolve()
original = root / 'pi05_libero_start/model.safetensors'
checkpoint = root / 'train-5steps/checkpoints/last/pretrained_model'
saved = checkpoint / 'model.safetensors'
state = json.loads((checkpoint.parent / 'training_state/training_step.json').read_text())
assert state['step'] == 5, state

comparisons = {}
with safe_open(original, framework='pt', device='cpu') as before, safe_open(saved, framework='pt', device='cpu') as after:
    assert len(before.keys()) == 812
    expected = {'model.' + key for key in before.keys()}
    assert expected <= set(after.keys())
    extra = set(after.keys()) - expected
    # The LeRobot model expands OpenPi's shared language embedding when saving.
    assert extra == {'model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'}, extra
    for key in ['action_in_proj.weight', 'action_out_proj.weight',
                'paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight']:
        new_key = 'model.' + key
        a, b = before.get_tensor(key).float(), after.get_tensor(new_key).float()
        delta = (a - b).abs()
        comparisons[key] = {'changed_elements': int((delta > 0).sum()),
                            'elements': a.numel(), 'max_abs_delta': float(delta.max())}
    assert comparisons['action_out_proj.weight']['changed_elements'] > 0
    assert comparisons['action_in_proj.weight']['changed_elements'] > 0
    assert comparisons['paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight']['changed_elements'] == 0

log = (root / 'train-5steps.log').read_text()
assert 'All keys loaded successfully!' in log
matches = re.findall(r'step:(\d+) .*?loss:([\d.]+) grdn:([\d.]+).*?mem_gb:([\d.]+)', log)
assert [int(row[0]) for row in matches] == list(range(1, 6)), matches
metrics = [{'step': int(s), 'loss': float(loss), 'gradient_norm': float(grad),
            'peak_allocated_memory_gib': float(mem)} for s, loss, grad, mem in matches]
assert all(np.isfinite(m['loss']) and np.isfinite(m['gradient_norm']) and m['gradient_norm'] > 0 for m in metrics)

print('Weight update checks:', comparisons, flush=True)
print('Reloading the saved five-step checkpoint...', flush=True)
policy, pre, post = load_policy(checkpoint)
dataset = LeRobotDataset('local/omy-short-check-v3', root=root / 'real_omy_v3',
                         delta_timestamps={'action': [i / 20 for i in range(50)]})
sample = dataset[40]  # First frame of held-out episode 2.
assert int(sample['episode_index']) == 2
policy.reset()
pre.reset()
post.reset()
torch.manual_seed(42)
batch = pre(sample)
round_trip = post(batch['action'].unsqueeze(0))
np.testing.assert_allclose(round_trip[0].cpu().numpy(), sample['action'].numpy(), atol=1e-5)
obs = {k: sample[k] for k in ['observation.images.image', 'observation.images.image2', 'observation.state', 'task']}
pre.reset()
post.reset()
start = time.perf_counter()
with torch.inference_mode():
    predicted = post(policy.predict_action_chunk(pre(obs)))
torch.cuda.synchronize()
inference_seconds = time.perf_counter() - start
assert tuple(predicted.shape) == (1, 50, 7), predicted.shape
assert torch.isfinite(predicted).all(), predicted
result = {
    'status': 'passed', 'host': 'Ubuntu workstation',
    'gpu': torch.cuda.get_device_name(0), 'source_weights': str(original.resolve()),
    'source_variant': 'official OpenPi pi05_libero_pytorch, BF16',
    'source_tensor_count': 812, 'saved_tensor_count': 813, 'complete_pretrained_load': True,
    'dataset': str(root / 'real_omy_v3'), 'collection': 'real MuJoCo OMY tiny joint-command trajectories, not successful grasp demonstrations',
    'total_frames': 60, 'train_frames': 40, 'train_episodes': [0, 1], 'heldout_episodes': [2],
    'fps': 20, 'state_semantics': 'six-dimensional EEF pose',
    'action_semantics': 'six absolute commanded joint targets plus normalized gripper',
    'steps': 5, 'batch_size': 1, 'chunk_size': 50, 'n_action_steps': 5,
    'train_expert_only': True, 'freeze_vision_encoder': True,
    'gradient_checkpointing': True, 'dtype': 'bfloat16',
    'metrics': metrics, 'parameter_update_checks': comparisons,
    'checkpoint': str(checkpoint), 'saved_training_step': state['step'],
    'checkpoint_reload': True, 'saved_processor_action_round_trip': True,
    'inference_shape': list(predicted.shape), 'inference_finite': True,
    'inference_seconds_first_call': inference_seconds,
    'grasp_success_evaluated': False,
}
(root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
(root / 'pi05_libero_start/model_provenance.json').write_text(json.dumps({
    'source_weights': str(original.resolve()), 'source_variant': result['source_variant'],
    'adapter': 'LeRobot 0.6.1 config and processors; original OpenPi tensor keys remapped by official PI05Policy loader',
    'tensor_count': 812, 'strict_pretrained_load': True,
    'note': 'Uses the cached LIBERO-finetuned pretrained policy, not generic lerobot/pi05_base weights.'
}, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
