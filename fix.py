import torch

checkpoint_path = '/u/rkie/gitco/vit-mae/mae_finetuned_vit_base.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

model_state_dict = checkpoint[‚model']

keys_to_remove = ['head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias']

for key in keys_to_remove:
    if key in model_state_dict:
        print(f"Lösche {key} aus dem Checkpoint...")
        del model_state_dict[key]

modified_checkpoint_path = '/u/rkie/gitco/vit-mae/mae_finetuned_vit_base.pth‘
checkpoint[‚model'] = model_state_dict
torch.save(checkpoint, modified_checkpoint_path)

