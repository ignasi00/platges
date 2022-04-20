
import copy
import re


def adapt_state_dict(state_dict):
    state_dict_v2 = copy.deepcopy(state_dict)

    for key in state_dict.keys():
        levels = key.split('.')

        new_key = re.sub(r'conv2_([0-9]+)', lambda match : f'pyconv_levels.{int(match.group(1)) - 1}', key)
        state_dict_v2[new_key] = state_dict_v2.pop(key)

        def replacement(match):
            names = ['conv1', 'bn1', 'bn1', 'bn1']
            return f'backbone.model.{names[int(match.group(1))]}.'
        new_key2 = re.sub(r'^layer0.([0-9]+).', replacement, new_key)
        state_dict_v2[new_key2] = state_dict_v2.pop(new_key)

        new_key = re.sub(r'^layer([1-9]+).', r'backbone.model.layer\1.', new_key2)
        state_dict_v2[new_key] = state_dict_v2.pop(new_key2)
    
    # There is no backbone.model.fc.weight and backbone.model.fc.bias => strict=False is mandatory
    
    return state_dict_v2
