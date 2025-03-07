import json

with open('data/api_mashup/raw/active_apis_data.txt', 'r', encoding='utf-8') as file:
    data = json.load(file)

for item in data:
    if item is None:
        continue
    if 'url' in item:
        del item['url']
    if 'versions' in item:
        del item['versions']
    if 'status' in item:
        del item['status']
    if 'style' in item:
        del item['style']

with open('data/api_mashup/raw/cleaned_apis_data.txt', 'w', encoding='utf-8') as file:
    json.dump(data, file)

print("处理完成，已删除所有 'url' 字段并保存为 'api_data_updated.json'。")