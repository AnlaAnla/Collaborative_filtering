import torch
from recommend_model import MatrixFactorization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    model = MatrixFactorization(num_users=8391 + 1, num_items=2016 + 1, embedding_dim=512)
    model.load_state_dict(torch.load('recommend_card01.pth'))
    model.to(device)

    # 获取用户333的embedding
    user_id = torch.tensor([0]).to(device)
    user_embedding = model.user_embeddings(user_id)

    # 计算所有电影的embedding与用户333的embedding的点积
    item_embeddings = model.item_embeddings.weight.to(device)
    dot_product = torch.matmul(user_embedding, item_embeddings.T)

    # 找到点积最大的5个电影
    top_k = torch.topk(dot_product, k=5)
    recommended_item_ids = top_k.indices.squeeze().tolist()

    # 打印推荐的电影
    print(f"为用户推荐的拼单编号为：{recommended_item_ids}")