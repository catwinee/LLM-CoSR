def remove_abnormal_nodes_with_llm(edge_index, node_descriptions, llm):
    # 构造提示词
    prompt = f"""
    你是一个图数据处理助手。给定一个图的边索引（edge_index）和每个节点的文字描述，请删除所有异常节点，并返回删除后的边索引和剩余节点的描述。

    输入：
    - edge_index: {edge_index}
    - 节点描述：
    """
    for node, desc in node_descriptions.items():
        prompt += f"  - 节点 {node}: {desc}\n"
    prompt += "\n输出："

    response = llm.generate(prompt)

    new_edge_index = parse_edge_index(response)
    remaining_nodes = parse_node_descriptions(response)

    return new_edge_index, remaining_nodes

def parse_edge_index(response):
    # 假设大模型的输出中包含 "删除后的 edge_index: [...]"
    start = response.find("删除后的 edge_index: [")
    end = response.find("]", start)
    edge_index_str = response[start:end+1]
    edge_index = eval(edge_index_str.split(": ")[1])
    return edge_index

def parse_node_descriptions(response):
    # 假设大模型的输出中包含 "剩余节点描述："
    start = response.find("剩余节点描述：")
    remaining_nodes = {}
    for line in response[start:].split("\n"):
        if line.strip().startswith("- 节点"):
            parts = line.split(": ")
            node = int(parts[0].split(" ")[1])
            desc = parts[1]
            remaining_nodes[node] = desc
    return remaining_nodes