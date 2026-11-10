import json
from math import atan, cosh, tanh
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qutai import Tensor, SafeTensors, concatenate, embedding, is_available, nn, save_safetensors, set_training


class PublicTensorTests(unittest.TestCase):
    def devices(self):
        return ['cpu', 'cuda'] if is_available('cuda') else ['cpu']

    def test_basic_slices_and_gradients(self):
        cases = [((slice(None), slice(None,None,-1)), (3,4), [3,2,1,0,7,6,5,4,11,10,9,8]),
                 ((-1, slice(1,None,2)), (2,), [9,11]),
                 ((None, Ellipsis, 1), (1,3), [1,5,9]),
                 ((slice(None,None,-2), None, slice(3,0,-2)), (2,1,2), [11,9,3,1]),
                 ((slice(0,0), Ellipsis), (0,4), []),
                 ((1,2), (), [6])]
        for device in self.devices():
            for dtype in ['float32', 'float64'] + (['bfloat16'] if device == 'cuda' else []):
                for index, shape, positions in cases:
                    with self.subTest(device=device,dtype=dtype,index=index):
                        x = Tensor(list(range(12)), shape=(3,4), requires_grad=True, device=device,dtype=dtype)
                        y = x[index]
                        self.assertEqual(y.shape,shape)
                        self.assertEqual(y.storage.tolist(),positions)
                        y.square().sum().backward(retain_graph=False)
                        self.assertEqual(x.grad.storage.tolist(),[2.*i if i in positions else 0. for i in range(12)])
                scalar = Tensor(3.,requires_grad=True,device=device,dtype=dtype)
                scalar[None,Ellipsis].sum().backward(retain_graph=False)
                self.assertEqual(scalar.grad.item(),1.)
                self.assertEqual(Tensor([],shape=(0,4),device=device,dtype=dtype)[::-1].shape,(0,4))
        x = Tensor([[1.,2.],[3.,4.]])
        for index, error in [(3,IndexError), ((Ellipsis,Ellipsis),IndexError), ((0,0,0),IndexError), (slice(None,None,0),ValueError), ([1],TypeError)]:
            with self.assertRaises(error): x[index]

    def test_split_concat_and_copy(self):
        for device in self.devices():
            x = Tensor(list(range(24)),shape=(2,3,4),requires_grad=True,device=device,dtype='float32')
            parts = x.split(2,axis=-1)
            self.assertEqual(concatenate(parts,axis=-1).tolist(),x.tolist())
            concatenate((x,x),axis=1).sum().backward(retain_graph=False)
            self.assertEqual(x.grad.storage.tolist(),[2.]*24)
            copy = Tensor([0.]*24,shape=x.shape,device=device,dtype=x.dtype)
            self.assertIs(copy.copy_(x),copy)
            self.assertEqual(copy.tolist(),x.tolist())
            self.assertEqual(x.split(2,axis=0)[1].shape,(1,3,4))
            for sections in (0,-1,3,1.5):
                with self.assertRaises(ValueError): x.split(sections)
            with self.assertRaises(ValueError): concatenate((x,Tensor([1.],device=device,dtype=x.dtype)),0)
            with self.assertRaises(ValueError): copy.copy_(Tensor(0.,device=device,dtype=x.dtype))
            with self.assertRaises(ValueError): concatenate(())

    def test_embedding_repeats_empty_and_gradients(self):
        for device in self.devices():
            for dtype in ['float32','float64'] + (['bfloat16'] if device=='cuda' else []):
                weight = Tensor(list(range(12)),shape=(3,4),requires_grad=True,device=device,dtype=dtype)
                out = embedding([[2,0],[2,1]],weight)
                self.assertEqual(out.shape,(2,2,4))
                self.assertEqual(out.storage.tolist(),list(range(8,12))+list(range(4))+list(range(8,12))+list(range(4,8)))
                out.sum().backward(retain_graph=False)
                self.assertEqual(weight.grad.storage.tolist(),[1.]*8+[2.]*4)
                self.assertEqual(embedding([],weight).shape,(0,4))
                with self.assertRaises(IndexError): embedding([3],weight)

    def test_cuda_does_not_expand_indices(self):
        if not is_available('cuda'): self.skipTest('需要 CUDA')
        from qutai.backends import cuda
        x = Tensor([1.]* (256*128),shape=(256,128),requires_grad=True,device='cuda')
        with patch.object(cuda,'gather',side_effect=AssertionError('逐元素索引路径被调用')):
            x[:,::2].sum().backward(retain_graph=False)
            self.assertEqual(embedding([1]*100,x).shape,(100,128))

    def test_unary_and_dropout(self):
        for device in self.devices():
            values = [-4.,-.2,0.,.7,5.]
            x = Tensor(values,requires_grad=True,device=device,dtype='float32')
            (x.tanh()+x.atan()).sum().backward(retain_graph=False)
            expected=[1/cosh(a)**2+1/(1+a*a) for a in values]
            self.assertLess(max(abs(a-b) for a,b in zip(x.grad.storage.tolist(),expected)),2e-6)
            self.assertLess(max(abs(a-b) for a,b in zip(x.tanh().tolist(),map(tanh,values))),2e-6)
            drop = nn.Dropout(.5,seed=3)
            with set_training(False): self.assertIs(drop(x),x)
            self.assertIs(drop(x,train=False),x)
            y=drop(Tensor([1.]*100,device=device,dtype='float32'))
            self.assertEqual(set(y.tolist()),{0.,2.})
            if device == 'cuda':
                self.assertEqual(set(drop(Tensor([1.]*100,device=device,dtype='bfloat16')).tolist()),{0.,2.})
        with TemporaryDirectory() as directory:
            path=Path(directory)/'drop.json'
            model=nn.Model([nn.Dropout(.5,seed=17)])
            model.save(path)
            expected=model(Tensor([1.]*20)).tolist()
            restored=nn.Model([]).load(path)
            self.assertEqual(restored(Tensor([1.]*20)).tolist(),expected)


class SafeTensorsTests(unittest.TestCase):
    def test_roundtrip_and_frozen_lazy_cache(self):
        for device in (['cpu','cuda'] if is_available('cuda') else ['cpu']):
            for dtype in ['float32','float64'] + (['bfloat16'] if device=='cuda' else []):
                with TemporaryDirectory() as directory:
                    path=Path(directory)/'model.safetensors'
                    source=Tensor([[.25,-3.5],[2.,7.]],requires_grad=True,device=device,dtype=dtype)
                    save_safetensors(path,{'weight':source,'empty':Tensor([],shape=(0,2),device=device,dtype=dtype),'scalar':Tensor(2.,device=device,dtype=dtype)},metadata={'test':'中文'})
                    data=path.read_bytes()
                    loaded=SafeTensors(path,device)
                    self.assertEqual(list(loaded),['weight','empty','scalar'])
                    self.assertEqual(loaded.metadata,{'test':'中文'})
                    self.assertFalse(loaded['weight'].requires_grad)
                    self.assertIs(loaded['weight'],loaded['weight'])
                    self.assertEqual(loaded['weight'].tolist(),source.tolist())
                    self.assertEqual(loaded['weight'].dtype,dtype)
                    self.assertEqual(loaded['empty'].shape,(0,2))
                    self.assertEqual(loaded['scalar'].item(),2.)
                    self.assertEqual(path.read_bytes(),data)
                    if dtype != 'bfloat16': self.assertEqual(SafeTensors(path)['weight'].tolist(),source.tolist())

    def test_invalid_files_and_metadata(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'bad.safetensors'
            for header,payload in [({'x':{'dtype':'F32','shape':[2],'data_offsets':[0,4]}},b'1234'),
                                   ({'x':{'dtype':'F32','shape':[-1],'data_offsets':[0,4]}},b'1234'),
                                   ({'x':{'dtype':'F32','shape':[1],'data_offsets':[-1,3]}},b'1234'),
                                   ({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]},'y':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},b'1234'),
                                   ({'__metadata__':{'key':4}},b'')]:
                encoded=json.dumps(header).encode()
                path.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
                with self.assertRaises(ValueError): SafeTensors(path)
            path.write_bytes(struct.pack('<Q',1000000000))
            with self.assertRaises(ValueError): SafeTensors(path)
            with self.assertRaises(ValueError): save_safetensors(path,{},metadata={'key':4})


if __name__ == '__main__':
    unittest.main()
