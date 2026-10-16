#!/usr/bin/env python3
"""Lawful single-goal language input; target geometry stays evaluator-only."""

ROOMS = {'bathroom', 'bedroom', 'kitchen', 'living room', 'office', 'hall',
         'laundry room', 'garage', 'dining room', 'rec room'}


def final_goal(task):
    objects = task.get('Object')
    regions = task.get('Region Name')
    if not isinstance(objects, list) or not isinstance(regions, list) or len(objects) != len(regions) or not objects:
        raise ValueError('missing or misaligned target labels')
    obj = str(objects[-1]).strip().lower()
    room = str(regions[-1]).strip().lower()
    if not obj or room not in ROOMS:
        raise ValueError(f'invalid final goal: object={obj!r}, room={room!r}')
    return {'goal_object': obj, 'goal_room': room,
            'goal_text': f'Target object: {obj}. Target room: {room}.'}


def policy_context(task):
    """Explicit goal language plus original instruction; no evaluator state."""
    return {'instruction': str(task.get('Task instruction') or ''),
            **final_goal(task), 'robot': str(task.get('Robot') or '').lower(),
            'level': len(task['Object'])}


if __name__ == '__main__':
    sample = {'Task instruction': 'cek target', 'Object': ['lamp', 'chair'],
              'Region Name': ['Bedroom', 'Living room'], 'Robot': 'stretch'}
    assert policy_context(sample)['goal_text'] == 'Target object: chair. Target room: living room.'
    sample['Region Name'][-1] = 'Tie'
    try:
        policy_context(sample)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid room must fail')
