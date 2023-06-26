import numpy as np


def get_dict():
    user_dict = {}
    item_dict = {}

    with open('card_user_item/user.data', 'r') as f:
        for line in f:
            user_id, user_code = line.strip().split("::")
            user_dict.update({int(user_code): int(user_id)})

    with open('card_user_item/item.data', 'r') as f:
        for line in f:
            item_id, item_code = line.strip().split("::")
            item_dict.update({int(item_code): int(item_id)})

    return user_dict, item_dict


if __name__ == '__main__':
    user_dict, item_dict = get_dict()
    train_data = open('card_user_item/rating_buy_train.data', 'w')

    with open('card_user_item/rating_buy.data', 'r') as f:

        for line in f:
            user_id, item_id, rating = line.strip().split('::')
            train_data.write("{}::{}::{}\n".format(user_dict[int(user_id)],
                                                   item_dict[int(item_id)],
                                                   rating))
    train_data.close()
    print('end')