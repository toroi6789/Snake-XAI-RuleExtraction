from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Đọc tệp Excel
file_path = "game_100_data.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Trích xuất các cột liên quan
df_selected = df[['Danger', 'Direction', 'Food direction', 'Action']]

# Mã hóa dữ liệu văn bản thành số
encoders = {col: LabelEncoder() for col in df_selected.columns}
df_encoded = df_selected.apply(lambda col: encoders[col.name].fit_transform(col))

# In ra bảng ánh xạ mã hóa cho từng cột
for column in ['Danger', 'Direction', 'Food direction', 'Action']:
    mapping = dict(zip(df_selected[column], df_encoded[column]))
    print(f"\nBảng mã hóa cho cột: {column}")
    for key, value in mapping.items():
        print(f"{key} -> {value}")

# Tạo cây quyết định
X = df_encoded[['Danger', 'Direction', 'Food direction']]
y = df_encoded['Action']
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf.fit(X, y)

# Xuất luật từ cây quyết định
tree_rules = export_text(clf, feature_names=['Danger', 'Direction', 'Food direction'])

# # Chuyển class số thành nhãn hành động gốc
# action_mapping = dict(zip(df_encoded['Action'], df_selected['Action']))
# for key, value in action_mapping.items():
#     tree_rules = tree_rules.replace(f"class: {key}", f"Action: {value}")

# In ra cây quyết định hoàn toàn bằng chữ
print(tree_rules)