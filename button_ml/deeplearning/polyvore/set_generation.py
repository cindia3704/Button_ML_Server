import json
import math
import os

import random
import tensorflow as tf
import pickle as pkl
import numpy as np
from . import configuration
from .import polyvore_model_bi as polyvore_model

# /home/buttonteam/Button_Server2/button/media/1605184420215_aINbjlZ.jpg
# serializer_data["photo"].replace("/home/buttonteam/Button_Server2/button/media/", ""):


def main(_):
    sample_bi_lstm_data = "images/media/201820205/5.jpg"
    sample_id = 3
    sample_style = "CASUAL"
    sample_season = "HWAN"
    set_generation(sample_bi_lstm_data, sample_id, sample_style, sample_season)


def set_generation(bi_lstm_input, id, style, season):
    g = tf.Graph()
    with g.as_default():
        model_config = configuration.ModelConfig()
        model = polyvore_model.PolyvoreModel(model_config, mode="inference")
        model.build()
        saver = tf.train.Saver()

        g.finalize()
        pkl_path = season + "_" + str(id) + ".pkl"
        #pkl_path = "HWAN_3.pkl"

        with tf.Session() as sess:
            #saver.restore(sess, "model/model_final/model.ckpt-34865")
            saver.restore(
                sess, "deeplearning/model/model_final/model.ckpt-34865")
            with open(pkl_path, "rb") as f:
                test_data = pkl.load(f)

            test_ids = list(test_data.keys())
            print("test_ids:")
            print(test_ids)
            test_feat = np.zeros((len(test_ids) + 1,
                                  len(test_data[test_ids[0]]["image_rnn_feat"])))
            test_emb = np.zeros((len(test_ids),
                                 len(test_data[test_ids[0]]["image_feat"])))

            for i, test_id in enumerate(test_ids):
                # Image feature in the RNN space.
                test_feat[i] = test_data[test_id]["image_rnn_feat"]
                # Image feature in the joint embedding space.
                test_emb[i] = test_data[test_id]["image_feat"]

            def norm_row(a):
                try:
                    return a / np.linalg.norm(a, axis=1)[:, np.newaxis]
                except:
                    return a / np.linalg.norm(a)

            test_emb = norm_row(test_emb)

            def rnn_one_step(sess, input_feed, lstm_state, direction='f'):
                if direction == 'f':
                    # Forward
                    [lstm_state, lstm_output] = sess.run(
                        fetches=['lstm/f_state:0',
                                 'f_logits/f_logits/BiasAdd:0'],
                        feed_dict={'lstm/f_input_feed:0': input_feed,
                                   'lstm/f_state_feed:0': lstm_state})
                else:
                    # Backward
                    [lstm_state, lstm_output] = sess.run(
                        fetches=['lstm/b_state:0',
                                 'b_logits/b_logits/BiasAdd:0'],
                        feed_dict={'lstm/b_input_feed:0': input_feed,
                                   'lstm/b_state_feed:0': lstm_state})

                return lstm_state, lstm_output

            def run_forward_rnn(sess, test_idx, test_feat, num_lstm_units):
                """ Run forward RNN given a query."""
                res_set = []
                lstm_state = np.zeros([1, 2 * num_lstm_units])
                for test_id in test_idx:
                    input_feed = np.reshape(test_feat[test_id], [1, -1])
                    # Run first step with all zeros initial state.
                    [lstm_state, lstm_output] = rnn_one_step(
                        sess, input_feed, lstm_state, direction='f')

                # Maximum length of the outfit is set to 3.
                for step in range(3):
                    curr_score = np.exp(
                        np.dot(lstm_output, np.transpose(test_feat)))
                    curr_score /= np.sum(curr_score)

                    next_image = np.argsort(-curr_score)[0][0]
                    # 0.00001 is used as a probablity threshold to stop the generation.
                    # i.e, if the prob of end-of-set is larger than 0.00001, then stop.
                    if next_image == test_feat.shape[0] - 1 or curr_score[0][-1] > 0.0001:
                        # print('OVER')
                        break
                    else:
                        input_feed = np.reshape(test_feat[next_image], [1, -1])
                        [lstm_state, lstm_output] = rnn_one_step(
                            sess, input_feed, lstm_state, direction='f')
                        res_set.append(next_image)

                return res_set

            def run_backward_rnn(sess, test_idx, test_feat, num_lstm_units):
                """ Run backward RNN given a query."""
                res_set = []
                lstm_state = np.zeros([1, 2 * num_lstm_units])
                for test_id in reversed(test_idx):
                    input_feed = np.reshape(test_feat[test_id], [1, -1])
                    [lstm_state, lstm_output] = rnn_one_step(
                        sess, input_feed, lstm_state, direction='b')
                for step in range(3):
                    curr_score = np.exp(
                        np.dot(lstm_output, np.transpose(test_feat)))
                    curr_score /= np.sum(curr_score)
                    next_image = np.argsort(-curr_score)[0][0]
                    # 0.00001 is used as a probablity threshold to stop the generation.
                    # i.e, if the prob of end-of-set is larger than 0.00001, then stop.
                    if next_image == test_feat.shape[0] - 1 or curr_score[0][-1] > 0.0001:
                        # print('OVER')
                        break
                    else:
                        input_feed = np.reshape(test_feat[next_image], [1, -1])
                        [lstm_state, lstm_output] = rnn_one_step(
                            sess, input_feed, lstm_state, direction='b')
                        res_set.append(next_image)

                return res_set

            def run_fill_rnn(sess, start_id, end_id, num_blank, test_feat, num_lstm_units):
                """Fill in the blanks between start and end."""
                if num_blank == 0:
                    return [start_id, end_id]
                lstm_f_outputs = []
                lstm_state = np.zeros([1, 2 * num_lstm_units])
                input_feed = np.reshape(test_feat[start_id], [1, -1])
                [lstm_state, lstm_output] = rnn_one_step(
                    sess, input_feed, lstm_state, direction='f')

                f_outputs = []
                for i in range(num_blank):
                    f_outputs.append(lstm_output[0])
                    curr_score = np.exp(
                        np.dot(lstm_output, np.transpose(test_feat)))
                    curr_score /= np.sum(curr_score)
                    next_image = np.argsort(-curr_score)[0][0]
                    input_feed = np.reshape(test_feat[next_image], [1, -1])
                    [lstm_state, lstm_output] = rnn_one_step(
                        sess, input_feed, lstm_state, direction='f')

                lstm_state = np.zeros([1, 2 * num_lstm_units])
                input_feed = np.reshape(test_feat[end_id], [1, -1])
                [lstm_state, lstm_output] = rnn_one_step(
                    sess, input_feed, lstm_state, direction='b')

                b_outputs = []
                for i in range(num_blank):
                    b_outputs.insert(0, lstm_output[0])
                    curr_score = np.exp(
                        np.dot(lstm_output, np.transpose(test_feat)))
                    curr_score /= np.sum(curr_score)
                    next_image = np.argsort(-curr_score)[0][0]
                    input_feed = np.reshape(test_feat[next_image], [1, -1])
                    [lstm_state, lstm_output] = rnn_one_step(
                        sess, input_feed, lstm_state, direction='b')

                outputs = np.asarray(f_outputs) + np.asarray(b_outputs)
                score = np.exp(np.dot(outputs, np.transpose(test_feat)))
                score /= np.sum(score, axis=1)[:, np.newaxis]
                blank_ids = np.argmax(score, axis=1)
                return [start_id] + list(blank_ids) + [end_id]

            def run_set_inference(sess, set_name, test_ids, test_feat, num_lstm_units):
                test_idx = []
                for name in set_name:
                    try:
                        test_idx.append(test_ids.index(name))
                    except:
                        print('not found')
                        return

                # dynamic search
                # run the whole bi-LSTM on the first item
                first_f_set = run_forward_rnn(
                    sess, test_idx[:1], test_feat, num_lstm_units)
                first_b_set = run_backward_rnn(
                    sess, test_idx[:1], test_feat, num_lstm_units)

                first_posi = len(first_b_set)
                first_set = first_b_set + test_idx[:1] + first_f_set

                image_set = []
                for i in first_set:
                    image_set.append(test_ids[i])

                if len(set_name) >= 2:
                    current_set = norm_row(test_feat[first_set, :])
                    all_position = [first_posi]
                    for test_id in test_idx[1:]:
                        # gradually adding items into it
                        # findng nn of the next item
                        insert_posi = np.argmax(
                            np.dot(norm_row(test_feat[test_id, :]), np.transpose(current_set)))
                        all_position.append(insert_posi)

                    # run bi LSTM to fill items between first item and this item
                    start_posi = np.min(all_position)
                    end_posi = np.max(all_position)

                    sets = run_fill_rnn(sess, test_idx[0], test_idx[1],
                                        end_posi - start_posi - 1, test_feat, num_lstm_units)

                else:
                    # run bi LSTM again
                    sets = test_idx
                f_set = run_forward_rnn(sess, sets, test_feat, num_lstm_units)
                b_set = run_backward_rnn(sess, sets, test_feat, num_lstm_units)

                image_set = []
                for i in b_set[::-1] + sets + f_set:
                    image_set.append(test_ids[i])

                return b_set[::-1] + sets + f_set

            def nn_search(i, test_emb, word_vec):
                # score = np.dot(test_emb, np.transpose(test_emb[i] + word_vec))
                score = np.dot(test_emb,
                               np.transpose(test_emb[i] + 2.0 * word_vec))
                return np.argmax(score)

            [word_emb] = sess.run([model.embedding_map])

            # Read word name
            # TO DO :: Final word dict 경로 넣기!!!!!!!!!!!!!!!!!!
            words = open(
                "deeplearning/final_word_dict.txt").read().splitlines()
            for i, w in enumerate(words):
                words[i] = w.split()[0]

            # Calculate the embedding of the word query
            # only run the first query for demo
            # set_name = 1605184420215_aINbjlZ.jpg

            # img_number = bi_lstm_input.replace("images/media", "").replace("201820205/", "").replace(".jpg", "").replace("/", "")
            # set_name = [str(str("201820205") + "_" + str(img_number))]
            set_name = [bi_lstm_input.replace]
            print(set_name)

            rnn_sets = run_set_inference(sess, set_name, test_ids,
                                         test_feat, model_config.num_lstm_units)

            print(rnn_sets)

            word_query = style

            formal_list = ['christian', 'marni', 'bags', 'clutch', 'classic', 'yoins',
                           'toe']
            semi_formal_list = ['flora', 'wool', 'gold', 'silk', 'flower', 'metallic', 'scarf', 'cashmere',
                                'wedding', 'skater', 'flare', 'shirts', 'necklace', 'suede']
            casual_list = ['classic', 'crop']

            outdoor_list = ['shorts', 'outdoor', 'nike',
                            'running', 'baseball', 'sports', 'short']
            vacance_list = ['summer', 'bikini', 'denim', 'shorts', 'sunglasses', 'floral', 'short', 'pants',
                            'island']

            if word_query == 'FORMAL':
                random_word_query = random.choice(formal_list)
            elif word_query == 'SEMI-FORMAL':
                random_word_query = random.choice(semi_formal_list)
            elif word_query == 'CASUAL':
                random_word_query = random.choice(casual_list)
            elif word_query == 'OUTDOOR':
                random_word_query = random.choice(outdoor_list)
            elif word_query == 'VACANCE':
                random_word_query = random.choice(vacance_list)
            else:
                print("잘못된 스타일")

            print(random_word_query)

            if random_word_query != "":
                # Get the indices of images.
                test_idx = []
                for name in set_name:
                    try:
                        test_idx.append(test_ids.index(name))
                    except:
                        print('not found')
                        return

                    # Calculate the word embedding
                random_word_query = [i + 1 for i in range(len(words))
                                     if words[i] in random_word_query.split()]
                # print(random_word_query)
                query_emb = norm_row(
                    np.sum(word_emb[random_word_query], axis=0))

                if style == "CASUAL":                              # 여기서 디폴트 처리 해야함 ㅠㅠ 이씽 ㅠㅠㅠㅠㅠ
                    for i, j in enumerate(rnn_sets):
                        if j in test_idx:                           # 원래 있던 결과랑 같으면 for문으로 다시 continue -> 다 돌아서 다 같은거 확인하면 random 라벨 재선택
                            continue
                        else:
                            rnn_sets[i] = nn_search(j, test_emb, query_emb)

                    random_word_query = random.choice(casual_list)
                    print(random_word_query)
                    if random_word_query != "":
                        test_idx = []
                        for name in set_name:
                            try:
                                test_idx.append(test_ids.index(name))
                            except:
                                print('not found')
                                return

                        random_word_query = [i + 1 for i in range(len(words))
                                             if words[i] in random_word_query.split()]
                        query_emb = norm_row(
                            np.sum(word_emb[random_word_query], axis=0))
                        for i, j in enumerate(rnn_sets):
                            if j not in test_idx:
                                rnn_sets[i] = nn_search(j, test_emb, query_emb)
                    print(rnn_sets)  # ㅂㅏ뀐 결과값

                else:
                    for i, j in enumerate(rnn_sets):
                        if j not in test_idx:
                            rnn_sets[i] = nn_search(j, test_emb, query_emb)
                    print(rnn_sets)

                # write images
            image_set = []
            for i in rnn_sets:
                image_set.append(test_ids[i])

            print(image_set)   # 최종 리턴
            return image_set
            # for i, image in enumerate(image_set):    result 안에 파일 넣어서 저장되는 부분
            #    name = image.split('_')
            #    os.system('cp %s/%s/%s.jpg %s/%d_%s.jpg' % ("images/media",
            #                                                name[0], name[1], "results", i, image))


if __name__ == "__main__":
    tf.app.run()
