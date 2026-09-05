import os, re

dir_path = r'C:\Users\chaya\Documents\Program\Practice\tensorflow'
files = [f for f in os.listdir(dir_path) if f.endswith('.py') and f.startswith('0')]

for f in files:
    filepath = os.path.join(dir_path, f)
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 1. UTF-8 Support
    if 'sys.stdout.reconfigure' not in content:
        content = content.replace('import os\n', 'import sys\nimport os\nif hasattr(sys.stdout, \"reconfigure\"):\n    try:\n        sys.stdout.reconfigure(encoding=\"utf-8\")\n    except Exception:\n        pass\n')
        
    # 2. Refactor Dataloaders
    # Find the create_windowed_dataset_pytorch definition
    def_regex = re.compile(r'def create_windowed_dataset_pytorch.*?return dataloader.*?$', re.MULTILINE | re.DOTALL)
    if def_regex.search(content):
        new_def = '''def create_windowed_tensors(X_data, y_data, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    return X_t, y_t, np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)'''
        content = def_regex.sub(new_def, content)
        
    # Remove the old DataLoader calls inside loop
    loop_regex = re.compile(r'    train_loader.*?create_windowed_dataset_pytorch.*?shuffle=True\)\n    val_loader.*?create_windowed_dataset_pytorch.*?shuffle=False\)\n    test_loader.*?create_windowed_dataset_pytorch.*?shuffle=False\)', re.MULTILINE | re.DOTALL)
    
    if loop_regex.search(content):
        new_loaders = '''    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)'''
        content = loop_regex.sub(new_loaders, content)
        
        # Insert pre-building block before the SEEDS loop
        prebuild_str = '''print('Pre-building sequence tensors...')
X_train_t, y_train_t, _, _ = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t, y_val_t, _, _     = create_windowed_tensors(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON)
X_test_t, y_test_t, X_test_seq, y_test_seq = create_windowed_tensors(X_test_scaled, y_test_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)
\nprint(f"Starting Automated'''
        content = content.replace('print(f"Starting Automated', prebuild_str)

    # 3. Informer
    if '02_' in f:
        content = re.sub(r'Q_reduce = torch.stack.*?view\(B, H, n_top, E\)', 'M_top_expanded = M_top.unsqueeze(-1).expand(-1, -1, -1, E)\n        Q_reduce = torch.gather(queries, 2, M_top_expanded)', content, flags=re.DOTALL)
        content = re.sub(r'for b in range\(B\):\n\s*for h in range\(H\):\n\s*context\[b, h, M_top\[b, h\], :\] = V_reduce\[b, h\]', 'M_top_expanded = M_top.unsqueeze(-1).expand(-1, -1, -1, D)\n        context.scatter_(2, M_top_expanded, V_reduce)', content)

    # 4. Autoformer
    if '03_' in f:
        content = re.sub(r'delays_agg\[b:b\+1\] = delays_agg\[b:b\+1\] \+ pattern\[b:b\+1\] \* tmp_values\[b:b\+1, :, idx:idx\+length, :\]', 'offsets = (index[:, i].unsqueeze(1) + time_seq).unsqueeze(1).unsqueeze(-1).expand(batch, head, length, channel)\n            sliced_values = torch.gather(tmp_values, 2, offsets)\n            delays_agg = delays_agg + pattern * sliced_values', content)
        content = re.sub(r'delays_agg = torch.zeros_like\(values\)', 'delays_agg = torch.zeros_like(values)\n        time_seq = torch.arange(length, device=values.device).unsqueeze(0)', content)
        content = re.sub(r'for b in range\(batch\):\n\s*idx = index\[b, i\].item\(\)\n\s*offsets', 'offsets', content)

    # 5. PatchTST
    if '05_' in f:
        content = re.sub(r'lookback = x\.size\(1\)\n\s*patches = \[\]\n\s*for i in range\(0, lookback - self\.patch_len \+ 1, self\.stride\):\n\s*p = x\[:, i : i \+ self\.patch_len, 0\] # \[batch \* features, patch_len\]\n\s*patches\.append\(p\)\n\s*patches = torch\.stack\(patches, dim=1\) # \[batch \* features, num_patches, patch_len\]', 'patches = x.squeeze(-1).unfold(1, self.patch_len, self.stride)', content, flags=re.DOTALL)

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f'Processed {f}')
