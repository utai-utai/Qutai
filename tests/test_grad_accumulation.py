import gc
import unittest
import weakref

from qutai import Tensor, is_available
from qutai.backends import cpu
from qutai.core import grad
from qutai.lm.recompute import checkpoint


class GradientAccumulationTests(unittest.TestCase):
    def test_leaf_sink_runs_before_remaining_backward_rules(self):
        x, y = Tensor(2, requires_grad=True), Tensor(3, requires_grad=True)

        def later(incoming):
            self.assertIsNotNone(x.grad)
            return (incoming,)

        middle = grad.record(Tensor(3), (y,), later)
        output = grad.record(Tensor(5), (middle, x), lambda g: (g, g))
        output.backward()
        self.assertEqual((x.grad.item(), y.grad.item()), (1, 1))

    def test_shared_parameters_and_repeated_backward_accumulate(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        loss = (x.square() + x.square()).sum()
        loss.backward()
        loss.backward()
        self.assertEqual(x.grad.tolist(), [8, 16, 24])
        (x * 3).sum().backward(retain_graph=False)
        self.assertEqual(x.grad.tolist(), [11, 19, 27])

    def test_seed_and_shared_rule_buffers_remain_independent(self):
        x = Tensor([1, 2], requires_grad=True)
        y = Tensor([3, 4], requires_grad=True)
        seed = Tensor([2, 3])
        output = grad.record(Tensor([4, 6]), (x, y), lambda g: (g, g))
        output.backward(seed, retain_graph=False)
        self.assertIsNot(x.grad.storage, seed.storage)
        self.assertIsNot(y.grad.storage, seed.storage)
        self.assertIsNot(x.grad.storage, y.grad.storage)
        x.backward([4, 5])
        self.assertEqual(x.grad.tolist(), [6, 8])
        self.assertEqual(y.grad.tolist(), [2, 3])
        self.assertEqual(seed.tolist(), [2, 3])

    def test_leaf_data_returned_by_rule_is_never_owned_as_gradient(self):
        x = Tensor([2, 3], requires_grad=True)
        output = grad.record(Tensor([1, 1]), (x,), lambda g: (x.storage,))
        output.backward([1, 1])
        output.backward([1, 1])
        self.assertEqual(x.tolist(), [2, 3])
        self.assertEqual(x.grad.tolist(), [4, 6])
        self.assertIsNot(x.grad.storage, x.storage)

    def test_existing_gradient_used_as_seed_is_detached_before_update(self):
        x = Tensor([1, 2], requires_grad=True)
        x.backward([2, 3])
        seed = x.grad
        (x + x).backward(seed)
        self.assertEqual(x.grad.tolist(), [6, 9])
        self.assertEqual(seed.tolist(), [2, 3])

    def test_existing_gradient_used_in_forward_is_not_mutated(self):
        x = Tensor([1, 2], requires_grad=True)
        x.backward([2, 3])
        previous = x.grad
        loss = (x * previous).sum()
        loss.backward()
        self.assertEqual(x.grad.tolist(), [4, 6])
        self.assertEqual(previous.tolist(), [2, 3])
        loss.backward()
        self.assertEqual(x.grad.tolist(), [6, 9])

    def test_user_assigned_shared_gradients_are_isolated(self):
        x = Tensor([1, 2], requires_grad=True)
        y = Tensor([3, 4], requires_grad=True)
        x.backward([2, 3])
        previous = x.grad
        y.grad = previous
        (x + y).sum().backward()
        self.assertEqual(x.grad.tolist(), [3, 4])
        self.assertEqual(y.grad.tolist(), [3, 4])
        self.assertEqual(previous.tolist(), [2, 3])
        self.assertIsNot(x.grad.storage, y.grad.storage)

    def test_gradient_tensor_shared_outside_current_graph_is_not_mutated(self):
        x = Tensor([1, 2], requires_grad=True)
        y = Tensor([3, 4], requires_grad=True)
        x.backward([2, 3])
        y.grad = x.grad
        (x * 2).sum().backward(retain_graph=False)
        self.assertEqual(x.grad.tolist(), [4, 5])
        self.assertEqual(y.grad.tolist(), [2, 3])

    def test_gradient_storage_shared_outside_current_graph_is_not_mutated(self):
        x = Tensor([1, 2], requires_grad=True)
        y = Tensor([3, 4], requires_grad=True)
        x.backward([2, 3])
        y.grad = Tensor(x.grad.storage)
        (x * 2).sum().backward(retain_graph=False)
        self.assertEqual(x.grad.tolist(), [4, 5])
        self.assertEqual(y.grad.tolist(), [2, 3])

    def test_zero_grad_drops_storage_and_unused_leaf_stays_none(self):
        x, unused = Tensor(2, requires_grad=True), Tensor(3, requires_grad=True)
        output = grad.record(Tensor(4), (x, unused), lambda g: (g, None))
        output.backward()
        reference = weakref.ref(x.grad.storage)
        self.assertIsNone(unused.grad)
        x.zero_grad()
        gc.collect()
        self.assertIsNone(reference())

    def test_default_engine_dictionary_and_checkpoint_contract_are_preserved(self):
        x, weight = Tensor(2, requires_grad=True), Tensor(3, requires_grad=True)
        output = x * weight + weight
        leaves = grad.backward(output, cpu.storage([1]))
        self.assertEqual(leaves[x].tolist(), [3])
        self.assertEqual(leaves[weight].tolist(), [3])
        self.assertIsNone(x.grad)
        self.assertIsNone(weight.grad)
        saved = checkpoint(lambda a: a * weight + weight, x, parameters=(weight,))
        saved.backward()
        saved.backward()
        self.assertEqual((x.grad.item(), weight.grad.item()), (6, 6))

    def test_released_graph_is_rejected_before_any_gradient_update(self):
        x = Tensor(2, requires_grad=True)
        output = x.square()
        output.backward(retain_graph=False)
        before = x.grad.item()
        with self.assertRaisesRegex(RuntimeError, "已释放"):
            output.backward()
        self.assertEqual(x.grad.item(), before)

    def test_micro_batch_does_not_keep_a_full_second_gradient_set(self):
        width, count = 32768, 12
        parameters = [Tensor(cpu.full(width), requires_grad=True) for _ in range(count)]

        contributions, live = [], []

        def rule(incoming):
            live.append(sum(reference() is not None for reference in contributions))
            values = cpu.full(width, 1)
            contributions.append(weakref.ref(values))
            return incoming, values

        def graph():
            output = Tensor(0)
            for parameter in parameters:
                output = grad.record(
                    Tensor(0),
                    (output, parameter),
                    rule,
                )
            return output

        graph().backward(retain_graph=False)
        contributions.clear()
        live.clear()
        graph().backward(retain_graph=False)
        streamed_live = max(live)
        contributions.clear()
        live.clear()
        leaves = grad.backward(graph(), cpu.storage([1]), retain_graph=False)
        self.assertEqual(len(leaves), count)
        self.assertLessEqual(streamed_live, 1)
        self.assertEqual(max(live), count - 1)
        for parameter in parameters:
            self.assertEqual(parameter.grad.storage[0], 2)
            self.assertEqual(parameter.grad.storage[-1], 2)


class CUDAGradientAccumulationTests(unittest.TestCase):
    def test_cuda_micro_batches_checkpoint_and_seed_alias(self):
        if not is_available("cuda"):
            self.skipTest("需要 CUDA 实卡")
        results = []
        for device in ("cpu", "cuda"):
            x = Tensor(
                [0.2, -0.7, 1.1], device=device, dtype="float32", requires_grad=True
            )
            weight = Tensor(
                [0.5, -0.3, 0.8], device=device, dtype="float32", requires_grad=True
            )
            for _ in range(3):
                output = checkpoint(
                    lambda a: (a * weight).square(), x, parameters=(weight,)
                )
                output.sum().backward(retain_graph=False)
            original = x.grad.storage
            seed = x.grad
            (x + x).backward(seed)
            self.assertIsNot(x.grad.storage, original)
            values = (x.grad.tolist(), weight.grad.tolist(), seed.tolist())
            (weight * 2).sum().backward(retain_graph=False)
            self.assertEqual(values[0], x.grad.tolist())
            results.append((*values, weight.grad.tolist()))
        for actual, expected in zip(results[1], results[0], strict=True):
            for value, reference in zip(actual, expected, strict=True):
                self.assertAlmostEqual(value, reference, places=5)


if __name__ == "__main__":
    unittest.main()
