import cv2
import sys
import time
import pymongo
import numpy as np
import os.path as osp

import torch

import _init_paths
from config import get_cfg_defaults


class FaceSingleDB:
    def __init__(self, db_name, fea_dim):
        self.cos_dis = torch.nn.CosineSimilarity()
        self.db_name = db_name
        face_client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
        self.face_db = face_client[self.db_name]
        self.person_features = None
        self.person_id_list = []
        self.face_features = None
        self.face_id_list = []
        self.fea_dim = fea_dim

        if 'person_col' in self.face_db.list_collection_names():
            self.person_id_max = self.face_db.person_col.find().sort('person_id', -1)[0]['person_id']
        else:
            self.person_id_max = 0

        if 'face_col' in self.face_db.list_collection_names():
            self.face_id_max = self.face_db.face_col.find().sort('face_id', -1)[0]['face_id']
        else:
            self.face_id_max = 0
        self.init_person_features()
        self.init_face_features()

    def insert_person(self, name, person_id=None):
        if person_id is None:
            self.person_id_max += 1
        else:
            assert person_id > self.person_id_max
            self.person_id_max = person_id
        feature = [0.0 for _ in range(self.fea_dim)]
        person = {'person_id': self.person_id_max, 'name': name,
                  'feature': feature, 'time': int(time.time())}
        self.face_db.person_col.insert_one(person)
        feature = torch.as_tensor(feature).cuda()
        if self.person_features is None:
            self.person_features = feature[None]
            self.person_id_list.append(self.person_id_max)
        else:
            self.person_features = torch.cat([self.person_features, feature[None]])
            self.person_id_list.append(self.person_id_max)

        return self.person_id_max

    def delete_person(self, person_id):
        res = self.face_db.person_col.delete_one({'person_id': person_id})
        if res.deleted_count:
            idx = self.person_id_list.index(person_id)
            self.person_features = self.person_features[torch.arange(self.person_features.size(0)) != idx]
            del self.person_id_list[idx]
            face_list = self.face_db.face_col.find({'person_id': person_id}, {'feature': 0})
            for face in face_list:
                self.delete_face(face['face_id'], update_person=False)

    def update_person(self, person_id, name=None):
        update_dict = {'time': int(time.time())}
        if name is not None:
            update_dict['name'] = name

        face_list = self.face_db.face_col.find({'person_id': person_id})
        feature_list = [torch.as_tensor(f['feature']) for f in face_list]
        feature_list = torch.stack(feature_list, dim=0)
        feature = torch.mean(feature_list, dim=0)
        update_dict['feature'] = feature.tolist()
        self.face_db.person_col.update_one({'person_id': person_id}, {'$set': update_dict})
        idx = self.person_id_list.index(person_id)
        self.person_features[idx] = feature.cuda()

    def query_person(self, feature, top_k=10):
        feature = torch.as_tensor(feature)[None].cuda()
        dis = self.cos_dis(feature, self.person_features)
        scores, inds = dis.topk(top_k)
        scores = scores.cpu().tolist()
        person_ids = []
        for i in inds:
            person_ids.append(self.person_id_list[i])
        return scores, person_ids

    def insert_face(self, person_id, feature, model=None, score=None):
        self.face_id_max += 1
        face = {'person_id': person_id, 'face_id': self.face_id_max, 'feature': feature,
                'time': int(time.time()), 'model': model, 'score': float(score)}
        self.face_db.face_col.insert_one(face)
        if self.face_features is None:
            self.face_features = torch.as_tensor(feature).cuda()[None]
            self.face_id_list.append(self.face_id_max)
        else:
            self.face_features = torch.cat([self.face_features, torch.as_tensor(feature).cuda()[None]])
            self.face_id_list.append(self.face_id_max)
        self.update_person(person_id)

    def delete_face(self, face_id, update_person=True):
        person_id = self.face_db.face_col.find({'face_id': face_id})[0]['person_id']
        res = self.face_db.face_col.delete_one({'face_id': face_id})
        if res.deleted_count:
            idx = self.face_id_list.index(face_id)
            self.face_features = self.face_features[torch.arange(self.face_features.size(0)) != idx]
            del self.face_id_list[idx]
            if update_person:
                self.update_person(person_id)

    def update_face(self, face_id, feature=None, model=None, score=None):
        update_dict = {'time': int(time.time())}
        if feature is not None:
            update_dict['feature'] = feature
            idx = self.face_id_list.index(face_id)
            self.face_features[idx] = feature
        if model is not None:
            update_dict['model'] = model
        if path is not None:
            update_dict['score'] = score
        self.face_db.face_col.update_one({'face_id': face_id}, {'$set': update_dict})
        person_id = self.face_db.face_col.find({'face_id': face_id})[0]['person_id']
        self.update_person(person_id)

    def query_face(self, feature, top_k=10):
        feature = torch.as_tensor(feature)[None].cuda()
        dis = self.cos_dis(feature, self.face_features)
        scores, inds = dis.topk(top_k)
        scores = scores.cpu().tolist()
        face_ids = []
        for i in inds:
            face_ids.append(self.face_id_list[i])
        return scores, face_ids

    def init_person_features(self):
        if 'person_col' in self.face_db.list_collection_names():
            all_person = self.face_db.person_col.find()
            features = []
            for person in all_person:
                person_id = person['person_id']
                self.person_id_list.append(person_id)
                if person['feature'] is None:
                    feature = [0.0 for _ in range(self.fea_dim)]
                else:
                    feature = person['feature']
                features.append(feature)
            self.person_features = torch.as_tensor(features).cuda()

    def init_face_features(self):
        if 'face_col' in self.face_db.list_collection_names():
            all_face = self.face_db.face_col.find()
            features = []
            for face in all_face:
                face_id = face['face_id']
                self.face_id_list.append(face_id)
                features.append(face['feature'])
            self.face_features = torch.as_tensor(features).cuda()

    def delete(self):
        self.face_db.person_col.drop()
        self.face_db.face_col.drop()
        self.person_id_max = 0
        self.face_id_max = 0
        self.person_features = None
        self.person_id_list = []
        self.face_features = None
        self.face_id_list = []


class FaceLib:
    def __init__(self):
        self.this_dir = osp.abspath(osp.dirname(__file__))
        yaml_path = osp.join(self.this_dir, 'face_lib.yaml')
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(yaml_path)
        self.cfg.freeze()

        if self.cfg.PET_ENGINE.USE_ENGINE:
            self.init_pet_engine()
        self.cos_dis = torch.nn.CosineSimilarity()

        self.face_db_dict = {}

    def init_pet_engine(self):
        pet_engine_path = osp.join(self.this_dir, self.cfg.PET_ENGINE.PATH)
        sys.path.append(pet_engine_path)
        from modules import pet_engine

        pet_engine_yaml_path = osp.join(self.this_dir, self.cfg.PET_ENGINE.CFG)
        module = pet_engine.MODULES['ObjectDet']
        self.det = module(cfg_file=pet_engine_yaml_path)
        module = pet_engine.MODULES['Face3DKpts']
        self.face_3dkpts = module()
        module = pet_engine.MODULES['FaceReco']
        self.face_reco = module(cfg_file=pet_engine_yaml_path)

    def face_det_reco(self, img, det_num=None, face_th=None):
        if det_num is None:
            det_num = self.cfg.PET_ENGINE.FACE_DET_NUM
        if face_th is None:
            face_th = self.cfg.PET_ENGINE.FACE_DET_TH
        output_det = self.det(img)
        im_labels = np.array(output_det['im_labels'])
        im_dets = output_det['im_dets'][im_labels == self.cfg.PET_ENGINE.FACE_DET_ID]
        boxes = im_dets[:, :4]
        scores = im_dets[:, -1]
        sorted_inds = np.argsort(-scores)
        boxes = boxes[sorted_inds[:det_num]]
        scores = scores[sorted_inds[:det_num]]

        results = []
        for i, score in enumerate(scores):
            if score > face_th:
                feature = self.face_reco(img, boxes[i:i+1])['features'][0].cpu().tolist()
                results.append({'boxes': boxes[i].tolist(), 'score': score, 'feature': feature})
        return results

    def face_comparison(self, feature1, feature2):
        feature1 = torch.as_tensor(feature1).cuda()
        feature2 = torch.as_tensor(feature2).cuda()
        dis = self.cos_dis(feature1[None], feature2[None])
        return float(dis[0].cpu())

    def create_face_db(self, face_db_name):
        if face_db_name in self.face_db_dict.keys():
            return '{} already exists'.format(face_db_name)
        self.face_db_dict[face_db_name] = FaceSingleDB(face_db_name, self.cfg.PET_ENGINE.FACE_FEA_DIM)

    def clean_face_db(self, face_db_name):
        self.face_db_dict[face_db_name].delete()

    def delete_face_db(self, face_db_name):
        self.face_db_dict[face_db_name].delete()
        del dict[face_db_name]

    def insert_person(self, face_db_name, name, person_id=None):
        return self.face_db_dict[face_db_name].insert_person(name, person_id=person_id)

    def delete_person(self, face_db_name, person_id):
        return self.face_db_dict[face_db_name].delete_person(person_id)

    def update_person(self, face_db_name, person_id, name=None):
        return self.face_db_dict[face_db_name].update_person(person_id, name=name)

    def query_person(self, face_db_name, feature, top_k=10):
        return self.face_db_dict[face_db_name].query_person(feature, top_k=top_k)

    def insert_face(self, face_db_name, person_id, feature, model=None, score=None):
        return self.face_db_dict[face_db_name].insert_face(person_id, feature, model=model, score=score)

    def delete_face(self, face_db_name, face_id, update_person=True):
        return self.face_db_dict[face_db_name].delete_face(face_id, update_person=update_person)

    def update_face(self, face_db_name, face_id, feature=None, model=None, score=None):
        return self.face_db_dict[face_db_name].update_face(face_id, feature=feature, model=model, score=score)

    def query_face(self, face_db_name, feature, top_k=10):
        return self.face_db_dict[face_db_name].query_face(feature, top_k=top_k)


def main():
    import glob
    face_lib = FaceLib()
    face_lib.create_face_db('star')
    face_lib.clean_face_db('star')
    # for name_temp in glob.glob('/home/user/Pictures/face_lib/spider_face/images/*'):
    #     name = name_temp.split('/')[-1]
    #     person_id = face_lib.face_db_dict['star'].insert_person(name)
    #     for image_name in glob.glob('/home/user/Pictures/face_lib/spider_face/images/{}/*'.format(name)):
    #         img = cv2.imread(image_name)
    #         results = face_lib.face_det_reco(img, det_num=1)
    #         if len(results):
    #             print('{}/{}'.format(name, image_name.split('/')[-1]))
    #             face_lib.face_db_dict['star'].insert_face(person_id, results[0]['feature'],
    #                                                       model='fcos_mv3', score=results[0]['score'])
    # for x in face_lib.face_db_dict['star'].face_db.person_col.find():
    #     print(x)

    img = cv2.imread('/home/user/Pictures/face_lib/000001.jpg')
    result = face_lib.face_det_reco(img)
    feature = result[0]['feature']

    names = ['陆毅', '万茜', '乔欣']
    for name in names:
        person_id = face_lib.face_db_dict['star'].insert_person(name)
        for image_name in glob.glob('/home/user/Pictures/face_lib/spider_face/images/{}/*'.format(name)):
            img = cv2.imread(image_name)
            results = face_lib.face_det_reco(img, det_num=1)
            if len(results):
                print('{}/{}'.format(name, image_name.split('/')[-1]))
                face_lib.insert_face('star', person_id, results[0]['feature'],
                                                          model='fcos_mv3', score=results[0]['score'])

    print(face_lib.query_person('star', feature, 2))
    print(face_lib.query_face('star', feature, 2))

    # for x in face_lib.face_db_dict['star'].face_db.face_col.find({}, {'feature': 0}):
    #     print(x)

    face_lib.delete_person('star', 1)

    # for x in face_lib.face_db_dict['star'].face_db.face_col.find({}, {'feature': 0}):
    #     print(x)

    print(face_lib.query_person('star', feature, 2))
    print(face_lib.query_face('star', feature, 2))


if __name__ == '__main__':
    main()

