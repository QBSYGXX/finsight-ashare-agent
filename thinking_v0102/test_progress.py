import unittest
from progress import Progress


class ProgressTests(unittest.TestCase):
    def test_no_heartbeat_during_input_and_exact_echo(self):
        messages = []
        monitor = Progress(output=lambda text, **kw: messages.append(text))
        def reader():
            for _ in range(10):
                self.assertFalse(monitor.tick())
            self.assertEqual(messages, ["财务指标> "])
            return "贵州茅台2025年毛利率同比增长率"
        self.assertEqual(monitor.read_question(reader), "贵州茅台2025年毛利率同比增长率")
        self.assertTrue(messages[-1].startswith("QUESTION="))
        self.assertTrue(monitor.tick())
        monitor.stop()

    def test_interrupted_input_restores_state(self):
        monitor = Progress(output=lambda *a, **kw: None)
        def reader():
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            monitor.read_question(reader)
        self.assertFalse(monitor.waiting)
        monitor.stop()

    def test_clock_counts_work_phase_not_user_waiting(self):
        now, messages = [0], []
        monitor = Progress(clock=lambda: now[0], output=lambda text, **kw: messages.append(text))
        now[0] = 200
        monitor.set_phase("load_model")
        now[0] = 230
        monitor.tick()
        self.assertEqual(messages[-1], "PROCESS_RUNNING phase=load_model phase_seconds=30")
        monitor.stop()
        self.assertFalse(monitor.tick())


if __name__ == "__main__":
    unittest.main()
