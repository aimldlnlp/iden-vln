"""Small, shared fixes for the isolated historical-policy repair experiment."""
import math


def signed_yaw(rotation):
    w, x, y, z = (float(getattr(rotation, field)) for field in ('w', 'x', 'y', 'z'))
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if not norm:
        raise ValueError('Zero quaternion')
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    return math.atan2(2*(w*y+x*z), 1-2*(y*y+z*z))


def before_action(trial, index, positions, yaws):
    index = int(index)
    assert index >= 1 and index < len(positions) and index < len(yaws)
    matches = [p for p in trial.iterdir() if p.is_dir() and p.name.split('_', 1)[0].isdigit()
               and int(p.name.split('_', 1)[0]) == index-1]
    assert len(matches) == 1, (trial, index, matches)
    return matches[0], positions[index-1], yaws[index-1]


def best_snapshot(model):
    # CPU detach alone still aliases parameters updated by the optimizer.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def self_check():
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    with tempfile.TemporaryDirectory() as folder:
        trial = Path(folder)
        (trial/'0_observe_target').mkdir()
        (trial/'1_turn_right_target').mkdir()
        frame, pose, yaw = before_action(trial, 1, [[0,0,0],[1,0,0]], [0,-0.5])
        assert frame.name == '0_observe_target' and pose == [0,0,0] and yaw == 0
    for angle in (-1.2, 0, 1.2):
        q = SimpleNamespace(w=math.cos(angle/2), x=0, y=math.sin(angle/2), z=0)
        assert abs(signed_yaw(q)-angle) < 1e-8
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.5))
    saved = best_snapshot(model)
    initial = saved['0.weight'].clone()
    with torch.no_grad():
        model[0].weight.add_(1)
    assert torch.equal(saved['0.weight'], initial)
    assert saved['0.weight'].data_ptr() != model[0].weight.data_ptr()
    model.eval()
    sample = torch.ones(2, 2)
    assert torch.equal(model(sample), model(sample))
    return {'pre_action': True, 'signed_yaw': True, 'independent_snapshot': True, 'eval_dropout': True}
