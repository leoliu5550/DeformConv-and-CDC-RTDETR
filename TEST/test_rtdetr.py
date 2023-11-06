import sys,pytest
sys.path.append(".")
import torch
import torch.nn as nn
from model.rtdetr import rtdetr
import dynamic_yaml


class Testrtdetr:

    with open('model_config.yaml') as file:
        cfg = dynamic_yaml.load(file)
    model = rtdetr('model_config.yaml').to(cfg.device)
    # @pytest. mark. skip(reason="dont test it now")
    def test_rtdetr(self):
        x =torch.ones([4,3,800,800]).to(self.cfg.device)
        out = self.model(x)

        
        print("#"*80)
        print("condee output")
        print(out.keys())
        print('pred_logits',out['pred_logits'].shape)
        print(out['pred_logits'])
        assert out['pred_logits'].shape == torch.Size([4, self.cfg.model.RTDETRTransformer.num_queries, 
                                        80])
        print('pred_boxes',out['pred_boxes'].shape)
        print(out['pred_boxes'])
        assert out['pred_boxes'].shape == torch.Size([4, self.cfg.model.RTDETRTransformer.num_queries, 4])



        print('#######aux_outputs######')
        assert len(out['aux_outputs']) ==6

        for item in out['aux_outputs']:
            print(item.keys(),item['pred_logits'].shape,item['pred_boxes'].shape)
        for item in out['aux_outputs']:
            print(item.keys(),item['pred_logits'].shape,item['pred_boxes'].shape)
            assert item['pred_logits'].shape == torch.Size([4, self.cfg.model.RTDETRTransformer.num_queries, 80])
            assert item['pred_boxes'].shape == torch.Size([4,self.cfg.model.RTDETRTransformer.num_queries, 4])