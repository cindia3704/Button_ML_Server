import tensorflow as tf
import pickle as pkl
import numpy as np
from . import configuration
from . import polyvore_model_bi as polyvore_model
import os
import json

# /home/buttonteam/Button_Server2/button/media/1605184420215_aINbjlZ.jpg


def main():
    sample_data = {"id": 22,
                   "clothID": 10,
                   "season": ["SUMMER", "WINTER"],
                   "category": "OUTER",
                   "photo": "/home/buttonteam/Button_Server2/button/media/1605184420215_aINbjlZ.jpg",
                   "style": ["SEMI-FORMAL", "CASUAL"],
                   "outfit": [6]
                   }
    extract_features(sample_data["id"], sample_data)


def extract_features(serializer_data):
    for one_season in serializer_data["season"]:  # json
        json_path = one_season + "_" + str(serializer_data["id"]) + ".json"

        if os.path.isfile(json_path):
            f = open(json_path, "r")
            json_dict = json.load(f)
            f.close()
            flag = False
            for one_item in json_dict[0]["items"]:
                if one_item["index"] == serializer_data["photo"].replace(
                        "/home/buttonteam/Button_Server2/button/media/", ""):
                    flag = True
            if flag:
                continue
            else:
                json_dict[0]["items"].append({
                    "index": serializer_data["photo"].replace("/home/buttonteam/Button_Server2/button/media/", "")
                })
                f2 = open(json_path, "w")
                json.dump(json_dict, f2)
                f2.close()
        else:
            output_data = []
            output_data.append(dict())
            output_data[0]["items"] = []
            output_data[0]["items"].append(dict())
            output_data[0]["items"][0]["index"] = serializer_data["photo"].replace(
                "/home/buttonteam/Button_Server2/button/media/", "")
            output_data[0]["set_id"] = "/home/buttonteam/Button_Server2/button/media/"
            f = open(json_path, "w")
            json.dump(output_data, f)
            f.close()

    g = tf.Graph()
    with g.as_default():
        model_config = configuration.ModelConfig()
        model_config.rnn_type = "lstm"
        model = polyvore_model.PolyvoreModel(model_config, mode="inference")
        model.build()
        saver = tf.train.Saver()

    g.finalize()
    sess = tf.Session(graph=g)
    saver.restore(sess, "deeplearning/model/model_final/model.ckpt-34865")

    for one_season in serializer_data["season"]:
        json_path = one_season + "_" + str(serializer_data["id"]) + ".json"
        pkl_path = one_season + "_" + str(serializer_data["id"]) + ".pkl"

        test_json = json.load(open(json_path))
        test_features = dict()

        if os.path.isfile(pkl_path):  # 있으면 append
            f = open(json_path, "r")
            json_dict = json.load(f)
            json_items = json_dict[0]["items"]
            number = len(json_items)
            append_from_number = number - 1
            f.close()
            print(append_from_number)

            added_test_features = dict()
            for image_set in test_json:
                set_id = image_set["set_id"]
                image_feat = []
                image_rnn_feat = []
                ids = []
            for image in image_set["items"][append_from_number:]:
                filename = os.path.join(set_id, str(image["index"]))
                print(filename)
                with tf.gfile.GFile(filename, "r") as f:
                    image_feed = f.read()

                    [feat, rnn_feat] = sess.run([model.image_embeddings,
                                                 model.rnn_image_embeddings],
                                                feed_dict={"image_feed:0": image_feed})

                    image_name = set_id + str(image["index"])
                    added_test_features[image_name] = dict()
                    added_test_features[image_name]["image_feat"] = np.squeeze(
                        feat)
                    added_test_features[image_name]["image_rnn_feat"] = np.squeeze(
                        rnn_feat)

                    print(added_test_features)

            f = open(pkl_path, "rb")
            data = pkl.load(f)
            data.update(added_test_features)
            f.close()

            f = open(pkl_path, "wb")
            pkl.dump(data, f)
            f.close()

        else:  # 없으면 새로 만들기
            k = 0
            for image_set in test_json:
                set_id = image_set["set_id"]
                image_feat = []
                image_rnn_feat = []
                ids = []
                k = k + 1
                print(str(k) + " : " + set_id)
                for image in image_set["items"]:
                    filename = os.path.join(set_id, str(image["index"]))
                    with tf.gfile.GFile(filename, "r") as f:
                        image_feed = f.read()

                    [feat, rnn_feat] = sess.run([model.image_embeddings,
                                                 model.rnn_image_embeddings],
                                                feed_dict={"image_feed:0": image_feed})

                    image_name = set_id + str(image["index"])
                    test_features[image_name] = dict()
                    test_features[image_name]["image_feat"] = np.squeeze(feat)
                    test_features[image_name]["image_rnn_feat"] = np.squeeze(
                        rnn_feat)

            with open(pkl_path, "wb") as f:
                pkl.dump(test_features, f)


def delete_extract_features(serializer_data):

    # 삭제할 옷이 json에 적혀있다면 -> 삭제 진행 / 없으면 pass
    for one_season in serializer_data["season"]:
        json_path = one_season + "_" + str(serializer_data["id"]) + ".json"

        if os.path.isfile(json_path):
            f = open(json_path, "r")
            json_dict = json.load(f)

            find_idx = -1
            for i, one_item in enumerate(json_dict[0]["items"]):
                if one_item["index"] == serializer_data["photo"].replace("/home/buttonteam/Button_Server2/button/media/", ""):
                    find_idx = i
                    break

            del (json_dict[0]["items"][find_idx])
            f2 = open(json_path, "w")
            json.dump(json_dict, f2)
            f2.close()

        else:
            pass

    for one_season in serializer_data["season"]:
        pkl_path = one_season + "_" + str(serializer_data["id"]) + ".pkl"

        if os.path.isfile(pkl_path):
            a = pkl.load(open(pkl_path, "rb"))   # type of a : dict

            search_key = None
            for one_media in a.keys():
                if one_media.replace("/home/buttonteam/Button_Server2/button/media/", "") == serializer_data["photo"].replace("/home/buttonteam/Button_Server2/button/media/", ""):
                    search_key = one_media
                    break

            if search_key is not None:
                del a[search_key]

            print(a.keys())

            f2 = open(pkl_path, "wb")
            a = pkl.dump(a, f2)
            f2.close()

        else:
            pass


def modify_extract_features(serializer_data):
    seasons = ["SUMMER", "WINTER", "HWAN"]

    for season in seasons:  # 모든 계절 json 다 확인하며 수정할 옷이 쓰여져 있으면 DELETE 진행
        json_path = season + "_" + str(serializer_data["id"]) + ".json"

        if os.path.isfile(json_path):
            f = open(json_path, "r")
            json_dict = json.load(f)

            find_idx = -1
            for i, one_item in enumerate(json_dict[0]["items"]):
                if one_item["index"] == serializer_data["photo"].replace("/home/buttonteam/Button_Server2/button/media/", ""):
                    find_idx = i
                    break
            if find_idx != -1:
                del (json_dict[0]["items"][find_idx])
                f2 = open(json_path, "w")
                json.dump(json_dict, f2)
                f2.close()
        else:
            pass

    for season in seasons:
        pkl_path = season + "_" + str(serializer_data["id"]) + ".pkl"

        if os.path.isfile(pkl_path):
            a = pkl.load(open(pkl_path, "rb"))  # type of a : dict

            search_key = None
            for one_media in a.keys():
                if one_media.replace("/home/buttonteam/Button_Server2/button/media/", "") == serializer_data["photo"].replace("/home/buttonteam/Button_Server2/button/media/", ""):
                    search_key = one_media
                    break

            if search_key is not None:
                del a[search_key]
                f2 = open(pkl_path, "wb")
                pkl.dump(a, f2)
                f2.close()
        else:
            pass

    for one_season in serializer_data["season"]:  # 수정된정보 다시 쓰
        json_path = one_season + "_" + str(serializer_data["id"]) + ".json"

        if os.path.isfile(json_path):
            f = open(json_path, "r")
            json_dict = json.load(f)
            f.close()
            flag = False
            for one_item in json_dict[0]["items"]:
                if one_item["index"] == serializer_data["photo"].replace("/home/buttonteam/Button_Server2/button/media/", ""):
                    flag = True
            if flag:
                continue
            else:
                json_dict[0]["items"].append({
                    "index": serializer_data["photo"].replace("/home/buttonteam/Button_Server2/button/media/", "")
                })
                f2 = open(json_path, "w")
                json.dump(json_dict, f2)
                f2.close()
        else:
            output_data = []
            output_data.append(dict())
            output_data[0]["items"] = []
            output_data[0]["items"].append(dict())
            output_data[0]["items"][0]["index"] = serializer_data["photo"].replace(
                "/home/buttonteam/Button_Server2/button/media/", "")
            output_data[0]["set_id"] = "/home / \
                buttonteam/Button_Server2/button/media/ "
            f = open(json_path, "w")
            json.dump(output_data, f)
            f.close()

    g = tf.Graph()
    with g.as_default():
        model_config = configuration.ModelConfig()
        model_config.rnn_type = "lstm"
        model = polyvore_model.PolyvoreModel(model_config, mode="inference")
        model.build()
        saver = tf.train.Saver()

    g.finalize()
    sess = tf.Session(graph=g)
    # TODO ::
    saver.restore(sess, "deeplearning/model/model_final/model.ckpt-34865")

    for one_season in serializer_data["season"]:
        json_path = one_season + "_" + str(serializer_data["id"]) + ".json"
        pkl_path = one_season + "_" + str(serializer_data["id"]) + ".pkl"

        test_json = json.load(open(json_path))
        test_features = dict()

        if os.path.isfile(pkl_path):  # 있으면 append
            f = open(json_path, "r")
            json_dict = json.load(f)
            json_items = json_dict[0]["items"]
            number = len(json_items)
            append_from_number = number - 1
            f.close()
            print(append_from_number)

            added_test_features = dict()
            for image_set in test_json:
                set_id = image_set["set_id"]
                image_feat = []
                image_rnn_feat = []
                ids = []
            for image in image_set["items"][append_from_number:]:
                filename = os.path.join(set_id, str(image["index"]))
                print(filename)
                with tf.gfile.GFile(filename, "r") as f:
                    image_feed = f.read()

                    [feat, rnn_feat] = sess.run([model.image_embeddings,
                                                 model.rnn_image_embeddings],
                                                feed_dict={"image_feed:0": image_feed})

                    image_name = set_id + str(image["index"])
                    added_test_features[image_name] = dict()
                    added_test_features[image_name]["image_feat"] = np.squeeze(
                        feat)
                    added_test_features[image_name]["image_rnn_feat"] = np.squeeze(
                        rnn_feat)

                    print(added_test_features)

            f = open(pkl_path, "rb")
            data = pkl.load(f)
            data.update(added_test_features)
            f.close()

            f = open(pkl_path, "wb")
            pkl.dump(data, f)
            f.close()

        else:  # 없으면 새로 만들기
            k = 0
            for image_set in test_json:
                set_id = image_set["set_id"]
                image_feat = []
                image_rnn_feat = []
                ids = []
                k = k + 1
                print(str(k) + " : " + set_id)
                for image in image_set["items"]:
                    filename = os.path.join(set_id, str(image["index"]))
                    with tf.gfile.GFile(filename, "r") as f:
                        image_feed = f.read()

                    [feat, rnn_feat] = sess.run([model.image_embeddings,
                                                 model.rnn_image_embeddings],
                                                feed_dict={"image_feed:0": image_feed})

                    image_name = set_id + str(image["index"])
                    test_features[image_name] = dict()
                    test_features[image_name]["image_feat"] = np.squeeze(feat)
                    test_features[image_name]["image_rnn_feat"] = np.squeeze(
                        rnn_feat)

            with open(pkl_path, "wb") as f:
                pkl.dump(test_features, f)
