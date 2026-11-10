from random import Random
import unittest
from unittest.mock import patch

from qutai import Tensor, divide, is_available, layer_norm, nn
from qutai.backends import cpu_lm, cuda
from qutai.backends.cuda_runtime import get_runtime


@unittest.skipUnless(is_available('cuda'), '需要 CUDA')
class ParallelRowsTests(unittest.TestCase):
    def close(self, actual, expected, tolerance=3e-5):
        self.assertEqual(len(actual),len(expected))
        self.assertLessEqual(max((abs(a-b)/max(1.,abs(b)) for a,b in zip(actual,expected)),default=0.),tolerance)

    def test_layer_norm_forward_and_all_gradients(self):
        random=Random(327)
        for dtype in ('float32','float64'):
            for rows,width in ((3,63),(5,64),(3,65),(2,1152),(2,2048)):
                values=[random.uniform(-2,2) for _ in range(rows*width)]
                weights=[random.uniform(.5,1.5) for _ in range(width)]
                biases=[random.uniform(-.5,.5) for _ in range(width)]
                probe=[random.uniform(-1,1) for _ in values]
                results=[]
                for device in ('cpu','cuda'):
                    x=Tensor(values,shape=(rows,width),requires_grad=True,device=device,dtype=dtype)
                    w=Tensor(weights,requires_grad=True,device=device,dtype=dtype)
                    b=Tensor(biases,requires_grad=True,device=device,dtype=dtype)
                    output=layer_norm(x,w,b)
                    (output*Tensor(probe,shape=x.shape,device=device,dtype=dtype)).sum().backward(retain_graph=False)
                    results.append([output.storage.tolist(),x.grad.storage.tolist(),w.grad.storage.tolist(),b.grad.storage.tolist()])
                for a,b in zip(*results): self.close(a,b,3e-5 if dtype=='float32' else 2e-11)

    def test_softmax_padding_arbitrary_width_and_axis(self):
        random=Random(98)
        for dtype in ('float32','float64'):
            for width,axis in ((63,-1),(64,-1),(65,-1),(423,-1),(1024,-1),(65,0)):
                values=[random.uniform(-20,20) for _ in range(3*width)]
                if axis==-1:
                    for row in range(3): values[row*width]= -float('inf')
                probe=[random.uniform(-1,1) for _ in values]
                results=[]
                for device in ('cpu','cuda'):
                    x=Tensor(values,shape=(3,width),requires_grad=True,device=device,dtype=dtype)
                    out=nn.Softmax(axis)(x)
                    (out*Tensor(probe,shape=x.shape,device=device,dtype=dtype)).sum().backward(retain_graph=False)
                    results.append([out.storage.tolist(),x.grad.storage.tolist()])
                for a,b in zip(*results):self.close(a,b,3e-5 if dtype=='float32' else 2e-11)

    def test_register_gemm_transposes_batches_and_accumulate(self):
        random=Random(123)
        rows,inner,cols=65,35,67
        for ta in (False,True):
            for tb in (False,True):
                a=[random.uniform(-1,1) for _ in range(rows*inner)]
                b=[random.uniform(-1,1) for _ in range(inner*cols)]
                left=Tensor(a,device='cuda',dtype='float32').storage
                right=Tensor(b,device='cuda',dtype='float32').storage
                cpu_a=Tensor(a,dtype='float64').storage
                cpu_b=Tensor(b,dtype='float64').storage
                reference=cpu_lm.gemm(cpu_a,cpu_b,rows,inner,cols,transpose_left=ta,transpose_right=tb).tolist()
                out=cuda.gemm(left,right,rows,inner,cols,batch=2,left_stride=0,right_stride=0,transpose_left=ta,transpose_right=tb)
                self.close(out.tolist(),reference*2)
                cuda.gemm(left,right,rows,inner,cols,batch=2,left_stride=0,right_stride=0,transpose_left=ta,transpose_right=tb,out=out,accumulate=True)
                self.close(out.tolist(),[2*x for x in reference]*2)

    def test_large_linear_training_gradients(self):
        random=Random(14)
        values=[random.uniform(-1,1) for _ in range(96*65)]
        weights=[random.uniform(-1,1) for _ in range(65*67)]
        results=[]
        for device in ('cpu','cuda'):
            x=Tensor(values,shape=(96,65),device=device,dtype='float32',requires_grad=True)
            w=Tensor(weights,shape=(65,67),device=device,dtype='float32',requires_grad=True)
            output=x@w
            output.square().mean().backward(retain_graph=False)
            results.append([output.storage.tolist(),x.grad.storage.tolist(),w.grad.storage.tolist()])
        for a,b in zip(*results):self.close(a,b)

    def test_metadata_reused_and_bounded(self):
        cuda._metadata.cache_clear()

        values=cuda.full(20,dtype='float32')
        with patch.object(cuda,'storage',wraps=cuda.storage) as upload:
            for _ in range(2):
                cuda.gather_slice(values,(2,5),(10,2),0)
                cuda.transpose(values,(4,5),(1,0))
            self.assertEqual(upload.call_count,2)
        for i in range(150):cuda._metadata(get_runtime(),(i,i+1))
        self.assertEqual(cuda._metadata.cache_info().currsize,128)
        cuda._metadata.cache_clear()

    def test_scalar_division_values_gradients_and_zero_checks(self):
        for dtype in ('float32','float64'):
            for divisor in (3.,-7.,1e-20,1e20):
                results=[]
                for scalar in (divisor,Tensor(divisor,device='cuda',dtype=dtype)):
                    x=Tensor([.3,-2.,1.],requires_grad=True,device='cuda',dtype=dtype)
                    y=divide(x,scalar)
                    y.sum().backward(retain_graph=False)
                    results.append((y.tolist(),x.grad.tolist()))
                for a,b in zip(*results):self.close(a,b,2e-6 if dtype=='float32' else 1e-12)
            x=Tensor([1.,2.],requires_grad=True,device='cuda',dtype=dtype)
            with patch.object(get_runtime(),'download',side_effect=AssertionError('常数除法不应同步读回 GPU')):
                (x/3.).sum().backward(retain_graph=False)
            for zero in (0.,-0.,Tensor(0.,device='cuda',dtype=dtype)):
                with self.assertRaises(ZeroDivisionError): x/zero
            self.assertEqual((Tensor([],device='cuda',dtype=dtype)/0.).tolist(),[])
        with self.assertRaises(ZeroDivisionError):Tensor([1.],device='cuda',dtype='float32')/1e-80
        self.assertEqual((Tensor([1.],device='cuda',dtype='float32')/float('inf')).item(),0.)



if __name__=='__main__':unittest.main()
