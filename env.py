import glob
import json
import os.path
import pickle
import shutil
from typing import Optional, List, Dict

import numpy as np
from sortedcontainers import SortedList
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from EnvSimulator.get_llm_resp import LLM, build_qa_prompt
from utils.match import is_matched_ctx_batch_llm_version, is_resp_match_answer_v2

# Event categories
TASK_GENERATE = 0

NETWORK_DELAY_RANGE = [0.02, 0.1]
TOP_K = 5
BATCH_SIZE = 64


def int_to_binary_array(n, bit_length=None):
    binary_str = bin(n)[2:]
    if bit_length:
        binary_str = binary_str.zfill(bit_length)
    binary_array = [int(bit) for bit in binary_str]
    return binary_array


class Env:
    def __init__(self, file_path, args,
                 qc_cache_path,
                 shard_path,
                 ):
        self.args = args
        self.rag_server_time_cost = simulate_rag_factory()
        self.qc_cache_path = qc_cache_path
        np.random.seed(42)
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
        self.last_finish_time: Optional[np.ndarray] = None
        self.events: Optional[SortedList] = None
        self.current_time: Optional[float] = None
        self.current_generated_task_data: Optional[Dict] = None
        self.transitions: Optional[List] = None

        file_paths = sorted(glob.glob(shard_path))
        self.data_on_shards = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f.readlines()]
            data = sorted(data, key=lambda d: d["id"])
            self.data_on_shards.append(data)

        if os.path.exists(self.qc_cache_path):
            with open(self.qc_cache_path, "rb") as f:
                self.qc_cache_dict = pickle.load(f)
        else:
            self.qc_cache_dict = {}
        self.llm = LLM()  # 谨记修改模型！！！

        model_path = "data/model_outputs/query_content_evaluator_v2/checkpoint-10000"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2, trust_remote_code=True, device_map="cuda:0"
        )
        self.model.eval()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.last_finish_time = np.zeros(self.args.num_shard)
        self.events = SortedList(key=lambda e: e[0])
        self.current_time = 0.
        self.current_generated_task_data = None
        self.transitions = []

        data = np.random.choice(self.data, size=2 + self.args.num_task_in_episode, replace=False)
        generate_time = 0
        for d in data:
            generate_time += max(np.random.normal(self.args.task_generate_interval_mean, 0.2 * self.args.task_generate_interval_mean), 0)
            self.events.add([generate_time, TASK_GENERATE, d])

    def process_event(self):
        event = self.events.pop(0)
        self.current_time = event[0]

        if event[1] == TASK_GENERATE:
            isScheduleRequired = True
            task_data = event[2]
            network_delays = np.random.uniform(low=NETWORK_DELAY_RANGE[0], high=NETWORK_DELAY_RANGE[1], size=(self.args.num_shard,))
            retrieval_time_cost = np.random.normal(self.rag_server_time_cost.get_mean(), self.rag_server_time_cost.get_std())
            state = np.hstack([task_data["question_embedding"], np.maximum(self.last_finish_time - self.current_time, 0), network_delays, retrieval_time_cost])
            self.current_generated_task_data = task_data
            return isScheduleRequired, state, retrieval_time_cost, network_delays

    def make_action(self, state, action, action_prob, retrieval_time_cost, network_delays):
        if len(self.transitions) != 0:
            self.transitions[-1]["next_state"] = state
            self.transitions[-1]["next_time"] = self.current_time
        if isinstance(action, list):
            server_selections = [int(a) for a in action]
        else:
            server_selections = int_to_binary_array(action, self.args.num_shard)

        valid_actions = [0 for _ in range(self.args.num_shard)]
        task_finish_time = self.current_time
        searched_copies = []
        for idx, isSelect in enumerate(server_selections):
            if not isSelect:
                continue
            if self.args.repeat_idx[idx] in searched_copies:
                continue
            tolerable_finish_time = self.current_time + self.args.tolerable_delay
            node_release_time = max(self.current_time, self.last_finish_time[idx])

            time_cost = retrieval_time_cost[idx] + network_delays[idx]
            self.last_finish_time[idx] = node_release_time + time_cost

            if node_release_time + time_cost <= tolerable_finish_time:
                valid_actions[idx] = 1
                searched_copies.append(self.args.repeat_idx[idx])

            task_finish_time = max(task_finish_time, min(node_release_time + time_cost, tolerable_finish_time))

        self.transitions.append({
            "task_data": self.current_generated_task_data,
            "current_time": self.current_time,
            "state": state,
            "action": server_selections,
            "action_prob": action_prob,
            "valid_action": valid_actions,
        })

    def process_transitions(self):
        for transition in self.transitions[1:]:
            task_data = transition["task_data"]
            question, answers, pos_idx = (
                task_data["question"], task_data["answers"], task_data["pos"]
            )

            contents = []
            for server_idx, isSelect in enumerate(transition["valid_action"]):
                if not isSelect:
                    continue
                assert self.data_on_shards[server_idx][pos_idx]["id"] == task_data["id"]
                sorted_shard_contents = sorted(self.data_on_shards[server_idx][pos_idx]["content"], key=lambda ctx: float(ctx["score"]), reverse=True)
                contents += sorted_shard_contents[:TOP_K]
            contents = [dict(t) for t in {frozenset(d.items()) for d in contents}]
            contents = sorted(contents, key=lambda ctx: float(ctx["score"]), reverse=True)[:TOP_K]

            transition["question"] = question
            transition["valid_contents"] = contents
            transition["prompt"] = build_qa_prompt(question, contents)

        self.transitions = self.transitions[1:-1]
        self.transitions, self.qc_cache_dict = is_matched_ctx_batch_llm_version(
            self.tokenizer, self.model,
            self.transitions, self.qc_cache_dict, prefix="valid"
        )

        self.transitions = self.llm.ask_model(self.transitions)

        processed_transitions = []
        for transition in self.transitions:
            is_correct = is_resp_match_answer_v2(transition["resp"], transition["task_data"]["answers"])
            reward = transition["valid_estimated_matched_score"]
            invalid_action_penalty = np.sum(np.array(transition["action"]) - np.array(transition["valid_action"]))

            if isinstance(transition["action"], list):
                transition["retrieve_overhead"] = np.sum(transition["action"])
            else:
                transition["retrieve_overhead"] = np.sum(int_to_binary_array(transition["action"], self.args.num_shard))
            reward -= (transition["retrieve_overhead"]) * self.args.weight
            processed_transitions.append({
                "current_time": transition["current_time"],
                "state": transition["state"],
                "action": transition["action"],
                "action_prob": transition["action_prob"],
                "reward": reward,
                "next_state": transition["next_state"],
                "next_time": transition["next_time"],
                "estimated_supportive_documents": transition["valid_estimated_matched_score"],
                "is_correct": is_correct,
                "task_data": transition["task_data"],
                "prompt": transition["prompt"],
                "resp":  transition["resp"],
                "retrieve_overhead": transition["retrieve_overhead"],
                "invalid_action_penalty": invalid_action_penalty,
            })
        self.transitions = sorted(processed_transitions, key=lambda pt: pt["current_time"])
        if np.random.randint(0, 4) == 1:
            self._save_cache_dict()

    def get_performance(self):
        reward_value_list = [t["reward"] for t in self.transitions]
        estimated_supportive_documents_list = [t["estimated_supportive_documents"] for t in self.transitions]
        is_valid_correct_list = [t["is_correct"] for t in self.transitions]
        retrieve_overhead = [t["retrieve_overhead"] for t in self.transitions]
        invalid_action_penalty = [t["invalid_action_penalty"] for t in self.transitions]
        return (
            np.average(reward_value_list),
            np.average(estimated_supportive_documents_list),
            np.average(is_valid_correct_list),
            np.average(retrieve_overhead),
            np.average(invalid_action_penalty)
        )

    def _save_cache_dict(self):
        with open("temp.bin", "wb") as f:
            pickle.dump(self.qc_cache_dict, f)
        shutil.copyfile("temp.bin", self.qc_cache_path)
        os.remove("temp.bin")
        print("Finish Saving")
