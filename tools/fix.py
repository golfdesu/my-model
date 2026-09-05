import os

dir_path = r'C:\Users\chaya\Documents\Program\Practice\tensorflow'
files = [f for f in os.listdir(dir_path) if f.endswith('.py') and f.startswith('0')]

block_to_move = '''print('Pre-building sequence tensors...')
X_train_t, y_train_t, _, _ = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t, y_val_t, _, _     = create_windowed_tensors(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON)
X_test_t, y_test_t, X_test_seq, y_test_seq = create_windowed_tensors(X_test_scaled, y_test_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)'''

for f in files:
    filepath = os.path.join(dir_path, f)
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    if block_to_move in content:
        content = content.replace(block_to_move, '')
        target_str = 'print(f\"Starting Automated'
        if target_str in content:
            content = content.replace(target_str, block_to_move + '\n\n' + target_str)
            
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f'Fixed {f}')
