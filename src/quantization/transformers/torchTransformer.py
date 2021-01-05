import copy
import inspect
import time
import types
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Log

class TorchTransformer(nn.Module):
    """!
    This class handle layer swap, summary, visualization of the input model
    """

    def __init__(self):
        super(TorchTransformer, self).__init__()

        self._register_dict = OrderedDict()
        self.log = Log()
        self._raw_TrochFuncs = OrderedDict()
        self._raw_TrochFunctionals = OrderedDict()

    # register class to trans
    def register(self, origin_class, target_class):
        """!
        This function register which class should transform to target class.		
        """
        print("register", origin_class, target_class)

        self._register_dict[origin_class] = target_class

        pass

    def trans_layers(self, model, update=True):
        """!
        This function transform layer by layers in register dictionarys

        @param model: input model to transfer

        @param update: default is True, wether to update the paramter from the orign layer or not. 
        Note that it will update matched parameters only.

        @return transfered model
        """
        # print("trans layer")
        if len(self._register_dict) == 0:
            print("No layer to swap")
            print(
                "Please use register( {origin_layer}, {target_layer} ) to register layer")
            return model
        else:
            for module_name in model._modules:
                # has children
                if len(model._modules[module_name]._modules) > 0:
                    self.trans_layers(model._modules[module_name])
                else:
                    if type(getattr(model, module_name)) in self._register_dict:
                        # use inspect.signature to know args and kwargs of __init__
                        _sig = inspect.signature(
                            type(getattr(model, module_name)))
                        _kwargs = {}
                        for key in _sig.parameters:
                            if _sig.parameters[key].default == inspect.Parameter.empty:  # args
                                # assign args
                                # default values should be handled more properly, unknown data type might be an issue
                                if 'kernel' in key:
                                    # _sig.parameters[key].replace(default=inspect.Parameter.empty, annotation=3)
                                    value = 3
                                elif 'channel' in key:
                                    # _sig.parameters[key].replace(default=inspect.Parameter.empty, annotation=32)
                                    value = 32
                                else:
                                    # _sig.parameters[key].replace(default=inspect.Parameter.empty, annotation=None)
                                    value = None

                                _kwargs[key] = value

                        _attr_dict = getattr(model, module_name).__dict__
                        _layer_new = self._register_dict[type(getattr(model, module_name))](
                            **_kwargs)  # only give positional args
                        _layer_new.__dict__.update(_attr_dict)

                        setattr(model, module_name, _layer_new)
        return model

    # torch.functionals
    def _torchFunctionals(self, raw_func, *args, **kwargs):
        """!
        The replaced torch.functional function (eg: F.{function}) will go here
        """
        # print("Functional")
        function_name = raw_func.__name__
        # print(raw_func.__name__)

        # functional has input expect affine_grid
        if function_name == "affine_grid":
            pass
        else:
            logs = args[0]
            cur_args = args[1:]

        # check is user used or in torch function call
        is_tensor_in = False
        # tensor input
        if (len(logs) > 1) and (type(logs[0]) == torch.Tensor):
            # print(logs[0].size(), logs[1].size())
            cur_inputs = logs
            is_tensor_in = True
            out = raw_func(*args, **kwargs)
            # print("Functional return : {}".format(out.size()))
            return raw_func(*args, **kwargs)

        elif (len(logs) == 1) and (type(logs) == torch.Tensor):
            cur_inputs = logs
            is_tensor_in = True
            out = raw_func(*args, **kwargs)
            # print("Functional return : {}".format(out.size()))
            return raw_func(*args, **kwargs)

        # log input
        else:
            # multi inputs
            bottoms = []
            cur_inputs = []
            if len(logs) > 1:
                cur_log = logs[0]
                for log in logs:
                    cur_inputs.append(log.cur_tensor)
                    bottoms.append(log.cur_id)
                    # update informations
                    cur_log.graph.update(log.graph)
                    cur_log.bottoms.update(log.bottoms)
                    cur_log.output_shape.update(log.output_shape)
                cur_inputs = tuple(cur_inputs)
            # one input
            else:
                cur_log = logs
                cur_inputs = cur_log.cur_tensor
                bottoms.append(cur_log.cur_id)

        # replace logs to tensor as function inputs to get output tensor
        args = list(args)
        args[0] = cur_inputs
        args = tuple(args)
        # send into origin functions
        #out_tensor = raw_func(*args, **kwargs).clone().detach()
        out_tensor = raw_func(*args, **kwargs).clone()

        # if function call, just return out tensor
        if is_tensor_in:
            return out_tensor

        # if log input and is function type, store as an layer
        if isinstance(raw_func, types.FunctionType):
            # use multiple address as name to prevent duplicate address
            layer_name = "F.{}_{}{}{}".format(
                function_name, id(out_tensor), id(args), id(kwargs))
            # replace with new address if still duplicate
            while layer_name in cur_log.graph:
                # if layer_name in cur_log.graph:
                # tmp_list = []
                # tmp_list.append(out_tensor)
                # tmp_tensor = copy.deepcopy(tmp_list[-1])
                # tmp_tensor = tmp_list[-1].clone()
                tmp_tensor = torch.tensor([0])

                # should not duplicate again?
                # layer_name = layer_name.split('.')[0] + "F" + ".{}_{}{}{}".format(function_name, id(tmp_tensor), id(args), id(kwargs))
                layer_name = "F.{}_{}{}{}{}".format(function_name, id(tmp_tensor), id(
                    args), id(kwargs), int((time.time()*100000) % 1000000))

            cur_log.graph[layer_name] = layer_name
            cur_log.bottoms[layer_name] = bottoms
            cur_log.cur_id = layer_name

        # if multi-output
        # if len(out_tensor) > 1:
        if not isinstance(out_tensor, torch.Tensor):
            out_logs = []
            for t in out_tensor:
                out_log = copy.deepcopy(cur_log)
                out_log.setTensor(t)
                out_logs.append(out_log)

            return out_logs
            return cur_log
