```python


def build_judge(**kwargs):
    from ...api import OpenAIWrapper, SiliconFlowAPI
    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    load_env()
    LOCAL_LLM = os.environ.get('LOCAL_LLM', None)
    if LOCAL_LLM is None:
        model_map = {
            'gpt-4-turbo': 'gpt-4-1106-preview',
            'gpt-4-0613': 'gpt-4-0613',
            'gpt-4-0125': 'gpt-4-0125-preview',
            'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
            'chatgpt-1106': 'gpt-3.5-turbo-1106',
            'chatgpt-0125': 'gpt-3.5-turbo-0125',
            'gpt-4o': 'gpt-4o-2024-05-13',
            'gpt-4o-0806': 'gpt-4o-2024-08-06',
            'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
            'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
            'deepseek': 'deepseek-ai/DeepSeek-V2.5',
        }
        model_version = model_map[model]
    else:
        model_version = LOCAL_LLM

    if model in ['qwen-7b', 'qwen-72b', 'deepseek']:
        model = SiliconFlowAPI(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model

import os
import re
import tempfile
from functools import partial
from jinja2.sandbox import SandboxedEnvironment
from jinja2 import Template

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich
import ipdb


class ImageVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'OCRVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv',
        'OCRVQA_TESTCORE': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv',
        'TextVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv',
        'DocVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv',
        'DocVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_TEST.tsv',
        'InfoVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv',
        'InfoVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_TEST.tsv',
        'ChartQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv',
        'GQA_TestDev_Balanced': 'https://opencompass.openxlab.space/utils/VLMEval/GQA_TestDev_Balanced.tsv',
    }

    DATASET_MD5 = {
        'OCRVQA_TEST': 'ca46a6d74b403e9d6c0b670f6fc00db9',
        'OCRVQA_TESTCORE': 'c5239fe77db8bdc1f2ad8e55e0d1fe97',
        'TextVQA_VAL': 'b233b31f551bbf4056f2f955da3a92cd',
        'DocVQA_VAL': 'd5ee77e1926ff10690d469c56b73eabf',
        'DocVQA_TEST': '6a2f28cac26ef2d3447374e8c6f6c8e9',
        'InfoVQA_VAL': '2342e9c225222f0ef4dec545ebb126fe',
        'InfoVQA_TEST': 'df535bf51b88dc9718252c34131a6227',
        'ChartQA_TEST': 'c902e0aa9be5582a7aad6dcf52734b42',
        'GQA_TestDev_Balanced': 'fead7df22befc1ed3ca2b62ea26fa17b',
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        # msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
        return msgs

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vqa_eval import hit_calculate, process_line

        data = load(eval_file)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        if listinstr(['TextVQA'], dataset):
            res = pool.map(partial(process_line, method='vqa_score'), lines)
        elif listinstr(['ChartQA'], dataset):
            res = pool.map(partial(process_line, method='relaxed_accuracy'), lines)
        elif listinstr(['OCRVQA', 'GQA'], dataset):
            res = pool.map(partial(process_line, method='accuracy'), lines)
        elif listinstr(['DocVQA', 'InfoVQA'], dataset):
            res = pool.map(partial(process_line, method='anls'), lines)
        else:  # default using vqa_score to calculate score
            res = pool.map(process_line, lines)
        hit = hit_calculate(res, dataset)
        ret = dict()
        if 'split' in data:
            splits = set(data['split'])
            for sp in splits:
                sub = [r for l, r in zip(lines, res) if l['split'] == sp]
                # [np.mean(x['match']) >= full_score_weight for x in sub]
                hit = hit_calculate(sub, dataset)
                ret[sp] = np.mean(hit) * 100
            sub = [r for l, r in zip(lines, res)]
            hit = hit_calculate(sub, dataset)
            ret['Overall'] = np.mean(hit) * 100
        else:
            ret['Overall'] = np.mean(hit) * 100
            if 'category' in data:
                cates = list(set(data['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    # [np.mean(x['match']) >= full_score_weight for x in sub]
                    hit = hit_calculate(sub, dataset)
                    ret[c] = np.mean(hit) * 100
        ret = d2df(ret)
        ret.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret




class OCRBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'OCRBench': 'https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv'
    }
    DATASET_MD5 = {'OCRBench': 'e953d98a987cc6e26ef717b61260b778'}

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        OCRBench_score = {
            'Regular Text Recognition': 0,
            'Irregular Text Recognition': 0,
            'Artistic Text Recognition': 0,
            'Handwriting Recognition': 0,
            'Digit String Recognition': 0,
            'Non-Semantic Text Recognition': 0,
            'Scene Text-centric VQA': 0,
            'Doc-oriented VQA': 0,
            'Key Information Extraction': 0,
            'Handwritten Mathematical Expression Recognition': 0,
        }

        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = str(line['prediction'])
            answers = eval(line['answer'])
            category = line['category']
            if category == 'Handwritten Mathematical Expression Recognition':
                for j in range(len(answers)):
                    answer = answers[j].strip().replace('\n', ' ').replace(' ', '')
                    predict = predict.strip().replace('\n', ' ').replace(' ', '')
                    if answer in predict:
                        OCRBench_score[category] += 1
                        break
            else:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace('\n', ' ')
                    predict = predict.lower().strip().replace('\n', ' ')
                    if answer in predict:
                        OCRBench_score[category] += 1
                        break

        final_score_dict = {}
        final_score_dict['Text Recognition'] = \
            (OCRBench_score['Regular Text Recognition'] + OCRBench_score['Irregular Text Recognition']
             + OCRBench_score['Artistic Text Recognition'] + OCRBench_score['Handwriting Recognition']
             + OCRBench_score['Digit String Recognition'] + OCRBench_score['Non-Semantic Text Recognition'])
        final_score_dict['Scene Text-centric VQA'] = OCRBench_score['Scene Text-centric VQA']
        final_score_dict['Doc-oriented VQA'] = OCRBench_score['Doc-oriented VQA']
        final_score_dict['Key Information Extraction'] = OCRBench_score['Key Information Extraction']
        final_score_dict['Handwritten Mathematical Expression Recognition'] = \
            (OCRBench_score['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score'] = \
            (final_score_dict['Text Recognition'] + final_score_dict['Scene Text-centric VQA']
             + final_score_dict['Doc-oriented VQA'] + final_score_dict['Key Information Extraction']
             + final_score_dict['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score Norm'] = (float(final_score_dict['Final Score']) / 10)
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class MathVista(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVista_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv'
    }
    DATASET_MD5 = {'MathVista_MINI': 'f199b98e178e5a2a20e7048f5dcb0464'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathvista import MathVista_auxeval, MathVista_acc

        model = judge_kwargs['model']
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MathVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVista_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = MathVista_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score


class MathVerse(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVerse_MINI': 'https://huggingface.co/datasets/CaraJ/Mathverse_VLMEvalKit/resolve/main/testmini.tsv', # noqa
        'MathVerse_MINI_Vision_Only': 'https://huggingface.co/datasets/CaraJ/Mathverse_VLMEvalKit/resolve/main/testmini_Vision_Only.tsv', # noqa
        'MathVerse_MINI_Vision_Dominant': 'https://huggingface.co/datasets/CaraJ/Mathverse_VLMEvalKit/resolve/main/testmini_Vision_Dominant.tsv', # noqa
        'MathVerse_MINI_Vision_Intensive': 'https://huggingface.co/datasets/CaraJ/Mathverse_VLMEvalKit/resolve/main/testmini_Vision_Intensive.tsv', # noqa
        'MathVerse_MINI_Text_Lite': 'https://huggingface.co/datasets/CaraJ/Mathverse_VLMEvalKit/resolve/main/testmini_Text_Lite.tsv', # noqa
        'MathVerse_MINI_Text_Dominant': 'https://huggingface.co/datasets/CaraJ/Mathverse_VLMEvalKit/resolve/main/testmini_Text_Dominant.tsv', # noqa
    }
    DATASET_MD5 = {
        'MathVerse_MINI': '5017caca32b7fa110c350a1bea861b65',
        'MathVerse_MINI_Vision_Only': '68a11d4680014ac881fa37adeadea3a4',
        'MathVerse_MINI_Vision_Dominant': 'b8fb63852d261ab2aaefba29cc2414d3',
        'MathVerse_MINI_Vision_Intensive': '01cbd35be202bb0c4873a4186a63bc19',
        'MathVerse_MINI_Text_Lite': '19e4b13bdd30b89a03b2e358bcfefa04',
        'MathVerse_MINI_Text_Dominant': '4f5cd2fa6630ea00bb11d6fde1f6fe6a',
    }

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathverse import MathVerse_auxeval_extract, MathVerse_auxeval_score, MathVerse_acc

        model = judge_kwargs['model']
        suffix = eval_file.split('.')[-1]
        storage_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.xlsx')
        tmp_file_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.pkl')
        storage_score = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file_score = eval_file.replace(f'.{suffix}', f'_{model}_score.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        # stage1: extract the answer
        if not osp.exists(storage_extract):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MathVerse evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_extract):
                ans = load(tmp_file_extract)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_extract,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_extract,
                )
                ans = load(tmp_file_extract)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_extract'] == v['log_extract'] and ans[k]['extract'] == v['extract']

            data['extract'] = [ans[idx]['extract'] for idx in data['index']]
            data['log_extract'] = [ans[idx]['log_extract'] for idx in data['index']]
            dump(data, storage_extract)

        # stage2: score the answer
        if not osp.exists(storage_score):
            data = load(storage_extract)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MathVerse evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_score):
                ans = load(tmp_file_score)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_score,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_score,
                )
                ans = load(tmp_file_score)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_score'] == v['log_score'] and ans[k]['score'] == v['score']

            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log_score'] = [ans[idx]['log_score'] for idx in data['index']]
            dump(data, storage_score)

        score = MathVerse_acc(storage_score)
        score_pth = storage_score.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score



class OlympiadBench(ImageBaseDataset):
    TYPE = 'VQA_ex_prompt'
    DATASET_URL = {
        'OlympiadBench': 'https://opencompass.openxlab.space/utils/VLMEval/OlympiadBench.tsv',
        'OlympiadBench_EN': 'https://opencompass.openxlab.space/utils/VLMEval/OlympiadBench_EN.tsv',
        'OlympiadBench_CN': 'https://opencompass.openxlab.space/utils/VLMEval/OlympiadBench_CN.tsv'
    }
    DATASET_MD5 = {
        'OlympiadBench': '9735ae0f0299eae1e7d07f5a7feab914',
        'OlympiadBench_EN': '5c68e100d394351fc7049f29d4d4efed',
        'OlympiadBench_CN': 'ea01b16788955702c79650c701e5b623'
    }

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        tgt_path_z = []
        if isinstance(line['image'], list):
            for i in range(len(line['image'])):
                tgt_path = osp.join(self.img_root, f"{line['index']}--{i+1}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'][i], tgt_path)
                tgt_path_z.append(tgt_path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path_z.append(tgt_path)
        return tgt_path_z

    def build_prompt(self, line):

        from .utils.olympiadbench import get_answer_type_text, make_input

        self.is_chinese = 'zh' in line['source']
        self.is_math = 'maths' in line['source']
        self.is_theorem_proving = 'TP' in line['source']

        if self.is_chinese:
            subject_content = '数学' if self.is_math else '物理'
            if self.is_theorem_proving:
                prompt = (
                    f"以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。"
                    "证明过程中使用的变量和公式请使用LaTeX格式表示。"
                )
            else:
                answer_type_text = get_answer_type_text(line['answer_type'], is_chinese=True,
                                                        multiple_answer=line['is_multiple_answer'])
                if line['is_multiple_answer']:
                    multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
                else:
                    multiple_answer_text = '\\boxed{答案}'
                unit_text = ''
                if line['unit']:
                    multiple_answer_text += '(单位)'
                    unit_text = '，注意答案的单位不要放在\\boxed{}中'
                prompt = (
                    f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。'
                    f'解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以“所以最终答案是{multiple_answer_text}。”'
                    f'显式给出结果{unit_text}。'
                )
        else:
            subject_content = 'Math' if self.is_math else 'Physics'
            if self.is_theorem_proving:
                prompt = (
                    f'The following is a theorem proving problem from an International {subject_content} competition. '
                    'Please use logical reasoning and common theorems to prove the proposition in the problem '
                    'according to the given requirements. '
                    'Please use LaTeX format to represent the variables and formulas used in the proof.'
                )
            else:
                if line['is_multiple_answer']:
                    multiple_answer_text = '\\boxed{multiple answers connected with commas}'
                else:
                    multiple_answer_text = '\\boxed{answer}'
                unit_text = ''
                if line['unit']:
                    multiple_answer_text += '(unit)'
                    unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
                answer_type_text = get_answer_type_text(line['answer_type'], is_chinese=False,
                                                        multiple_answer=line['is_multiple_answer'])
                prompt = (
                    f'The following is an open-ended problem from an International {subject_content} competition. '
                    f'{answer_type_text}Please calculate the answer according to the given requirements and '
                    'the information provided. Please use LaTeX format to represent the variables and formulas '
                    'used in the solution process and results. Please end your solution with "So the final answer '
                    f'is {multiple_answer_text}." and give the result explicitly{unit_text}.'
                )

        if self.is_math:
            input = make_input(prompt, line['question'])
        else:
            if 'context' in line.keys() and str(line['context']) != 'nan':  # cannot be null
                input = make_input(prompt, line['context'] + '\n' + line['question'])
            else:
                input = make_input(prompt, line['question'])

        ret = [dict(type='text', value=input)]
        tgt_path = self.dump_image(line)

        ret.extend([dict(type='image', value=s) for s in tgt_path])

        return ret

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.olympiadbench import MathJudger, extract_answer
        judger = MathJudger()

        suffix = eval_file.split('.')[-1]
        name_str1 = 'judge'
        name_str2 = 'score'
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str1}_result.xlsx')
        score_file = eval_file.replace(f'.{suffix}', f'_{name_str2}_result.csv')

        if not osp.exists(result_file):
            data = load(eval_file)
            scorez = []

            for i in tqdm(data.iterrows()):
                line = i[1]
                model_answer = line['prediction']
                is_chinese = 'zh' in line['source']
                model_answer = extract_answer(is_chinese, model_answer, is_deepseek=False)
                answer_type = line['answer_type']

                final_answer = line['final_answer'][2:-2]

                if str(answer_type) != 'nan' and 'Tuple' in answer_type:
                    judge_result = judger.judge(model_answer, final_answer)
                else:
                    if str(line['error']) != 'nan':
                        if ',' in line['error']:
                            precisions = line['error'].split(',')
                            precisions = [float(p) if p else 1e-8 for p in precisions]
                            judge_result = judger.judge(model_answer, final_answer, precisions)
                        else:
                            precision = float(line['error'])
                            judge_result = judger.judge(model_answer, final_answer, precision)
                    else:
                        judge_result = judger.judge(model_answer, final_answer)
                scorez.append(judge_result)

            data['score'] = scorez
            dump(data, result_file)

        judge_file = load(result_file)

        if not osp.exists(score_file):
            name_list = ['OE_MM_maths_en_COMP', 'OE_MM_maths_zh_CEE', 'OE_MM_maths_zh_COMP', 'OE_MM_physics_en_COMP',
                         'OE_MM_physics_zh_CEE','OE_TO_maths_en_COMP', 'OE_TO_maths_zh_CEE', 'OE_TO_maths_zh_COMP',
                         'OE_TO_physics_en_COMP', 'OE_TO_physics_zh_CEE']

            sample_list = [[] for _ in range(len(name_list))]
            for i in judge_file.iterrows():
                line = i[1]
                for j in range(len(name_list)):
                    if line['source'] == name_list[j]:
                        sample_list[j].append(line['score'])

            acc_dict = {}
            correct_list = []

            # fine-grained
            for i in range(len(name_list)):
                correct_num = 0
                for j in sample_list[i]:
                    if j:
                        correct_num += 1
                correct_list.append(correct_num)
                acc = 100 * correct_num / len(sample_list[i])
                acc_dict[name_list[i]] = [acc]

            # 4 grained
            labela = ['zh', 'en']
            labelb = ['maths', 'physics']

            grain_list = [[x,y] for x in labela for y in labelb]
            for j in grain_list:
                dict_name = j[0] + "_" + j[1]
                correct_num = 0
                full_num = 0
                for i in range(len(name_list)):
                    if all(k in name_list[i] for k in j):
                        correct_num += correct_list[i]
                        full_num += len(sample_list[i])
                acc = 100 * correct_num / full_num
                acc_dict[dict_name] = [acc]

            # 2 grained
            grain_list = ['maths', 'physics']
            for j in grain_list:
                dict_name = j
                correct_num = 0
                full_num = 0
                for i in range(len(name_list)):
                    if j in name_list[i]:
                        correct_num += correct_list[i]
                        full_num += len(sample_list[i])
                acc = 100 * correct_num / full_num
                acc_dict[dict_name] = [acc]

            # AVG
            correct_num = sum(correct_list)
            acc = 100 * correct_num / len(judge_file)
            acc_dict['AVG'] = [acc]

            acc_pd = pd.DataFrame(acc_dict)
            acc_pd.to_csv(score_file, index=False, encoding='gbk')

        accdz = pd.read_csv(score_file)
        return accdz

class MMVet(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMVet': 'https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv'
    }
    DATASET_MD5 = {'MMVet': '748aa6d4aa9d4de798306a63718455e3'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mmvet import MMVet_auxeval, MMVet_acc

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=3, **judge_kwargs)
            assert model.working(), ('MMVet evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMVet_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v['score']
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMVet_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        score_fine_pth = storage.replace('.xlsx', '_score_fine.csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score


class CustomVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError


class CRPE(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'CRPE_EXIST': 'https://huggingface.co/datasets/petter12321/crpe_vlmevalkit/resolve/main/CRPE_EXIST.tsv',
        'CRPE_RELATION': 'https://huggingface.co/datasets/petter12321/crpe_vlmevalkit/resolve/main/CRPE_RELATION.tsv'
    }
    DATASET_MD5 = {
        'CRPE_EXIST': '315584e23ac1ff7f8719ed3b7ad90f08',
        'CRPE_RELATION': 'bad7094cde0b572288f4b119c2d0c656'}

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.crpe import is_correct
        # find-image, count-text, find-text,
        # infer-choose, count-image, visual-reasoning
        score = {
            'exist': 0,
            'subject': 0,
            'predicate': 0,
            'object': 0,
            'total': 0,
        }
        num = {
            'exist': 0,
            'subject': 0,
            'predicate': 0,
            'object': 0,
            'total': 0,
        }
        final_score_dict = {
            'exist': 0,
            'subject': 0,
            'predicate': 0,
            'object': 0,
            'total': 0,
        }
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = str(line['prediction'])
            answers = str(line['answer'])
            # print("predict =", predict)
            # print("answers =", answers)
            category = line['category']
            if is_correct(answers, predict):
                score[category] += 1
                score['total'] += 1
            num[category] += 1
            num['total'] += 1

        for category in ['exist', 'subject', 'predicate', 'object', 'total']:
            if num[category] != 0:
                final_score_dict[category] = score[category] / num[category]
            else:
                final_score_dict[category] = None

        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict

    def build_prompt(self, line):
        ROOT = LMUDataRoot()
        msgs = super().build_prompt(line)
        for msg in msgs:
            if msg['type'] == 'image':
                msg['value'] = osp.join(osp.join(ROOT, 'images', self.dataset_name), msg['value'])
        return msgs


class QSpatial(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'QSpatial_plus': '',
        'QSpatial_scannet': ''
    }

    # NOTE: To evaluate Q-Spatial-ScanNet, you need to get the permission from ScanNet website
    # Once you get the permission, you can use the helper code here to download and extract necessary images:
    # https://github.com/andrewliao11/Q-Spatial-Bench-code?tab=readme-ov-file#for-qspatial_scannet
    qspatial_root = "TO_BE_REPLACED_WITH_THE_PATH_TO_QSPATIAL_DATASET"
    url = "https://raw.githubusercontent.com/andrewliao11/Q-Spatial-Bench-code/refs/heads/main/prompt_templates/"

    def post_build(self, dataset):
        # Download the prompt templates from github

        links = [
            self.url + "system_prompt.txt",
            self.url + "spatial_prompt_single.txt",
            self.url + "spatial_prompt_steps.txt",
            self.url + "standard_prompt.txt",
            self.url + "zero_shot_prompt.txt"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for link in links:
                tgt_path = os.path.join(temp_dir, link.split("/")[-1])
                os.system(f"wget {link} -O {tgt_path}")

            self.system_prompt = open(os.path.join(temp_dir, "system_prompt.txt")).read()
            self._prompt_templates = dict(
                spatial_prompt_single=open(os.path.join(temp_dir, "spatial_prompt_single.txt")).read(),
                spatial_prompt_steps=open(os.path.join(temp_dir, "spatial_prompt_steps.txt")).read(),
                standard_prompt=open(os.path.join(temp_dir, "standard_prompt.txt")).read(),
                zero_shot_prompt=open(os.path.join(temp_dir, "zero_shot_prompt.txt")).read(),
            )

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):

        text_prompt_template = self._prompt_templates["spatial_prompt_single"]
        env = SandboxedEnvironment()
        text_prompt = env.from_string(text_prompt_template).render(question=line["question"])
        tgt_path = self.dump_image(line)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        msgs.append(dict(type='text', value=f"{self.system_prompt}\n{text_prompt}"))
        return msgs

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        import io
        import pandas as pd
        from datasets import load_dataset

        hf_dataset = load_dataset("andrewliao11/Q-Spatial-Bench", split=dataset)
        df = hf_dataset.to_pandas()

        df.reset_index(drop=True, inplace=True)
        df['index'] = df.index
        df['answer'] = list(zip(df['answer_value'], df['answer_unit']))
        df = df[['index'] + [col for col in df.columns if col != 'index']]

        if dataset == "QSpatial_scannet":
            df = df.drop(columns=["image"])
            df["image"] = [Image.open(os.path.join(self.qspatial_root, image_path)) for image_path in df["image_path"]]
        else:
            df["image"] = [Image.open(io.BytesIO(image_dict["bytes"])) for image_dict in df["image"]]

        df["image"] = [encode_image_to_base64(image) for image in df["image"]]
        return df

    @classmethod
    def get_multiplier(self, unit):

        unit = unit.lower()
        if unit in ["meters", "meter", "m", "metre", "metres"]:
            multiplier = 100
        elif unit in ["centimeters", "centimeter", "cm"]:
            multiplier = 1
        elif unit in ["feet", "foot", "ft"]:
            multiplier = 30.48
        elif unit in ["inch", "inches", "in"]:
            multiplier = 2.54
        elif unit in ["mm"]:
            multiplier = 0.1
        else:
            print(f"Unknown unit: {unit}")
            multiplier = 0.

        return multiplier

    @classmethod
    def parse_string(self, input_str):
        # Regular expression to match the pattern (number or range, text)
        match = re.match(r'\(([\d.-]+), (.+)\)', input_str)
        if match:
            number_part = match.group(1)
            text = match.group(2)

            if '-' in number_part:
                start, end = map(float, number_part.split('-'))
                number = (start + end) / 2
            else:
                number = float(number_part)

            return number * self.get_multiplier(text)
        else:
            print(f"Unable to parse the input string {input_str}")
            return 0

    @classmethod
    def parse_prediction(self, vlm_response):
        # Value
        pattern = r'scalar{([^}]*)}'
        str_inside_scalar_boxes = re.findall(pattern, vlm_response)[-1]
        scalar_list = re.findall(r'\d+\.?\d*', str_inside_scalar_boxes)
        parsed_scalar = np.array(scalar_list).astype(float).mean()

        # Unit
        pattern = r'distance_unit{([^}]*)}'
        str_inside_unit_boxes = re.findall(pattern, vlm_response)
        parsed_unit = str_inside_unit_boxes[-1]

        pred_value_in_cms = parsed_scalar * self.get_multiplier(parsed_unit)
        return pred_value_in_cms

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        data = load(eval_file)
        if "model" in judge_kwargs:
            from .utils.qspatial import QSpatial_auxeval

            # extract using model
            model = judge_kwargs['model']
            suffix = eval_file.split('.')[-1]
            storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
            tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
            nproc = judge_kwargs.pop('nproc', 4)

            if not osp.exists(storage):
                model = build_judge(max_tokens=128, **judge_kwargs)

                assert model.working(), ('Evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
                lt = len(data)
                lines = [data.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = [line['index'] for line in lines]

                ans = {}
                if osp.exists(tmp_file):
                    ans = load(tmp_file)
                tups = [x for x, i in zip(tups, indices) if i not in ans]
                indices = [i for i in indices if i not in ans]

                if len(indices):
                    new_results = track_progress_rich(
                        QSpatial_auxeval,
                        tups,
                        nproc=nproc,
                        chunksize=nproc,
                        keys=indices,
                        save=tmp_file,
                    )
                    ans = load(tmp_file)
                    for k, v in zip(indices, new_results):
                        assert k in ans
                        assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

                data['res'] = [ans[idx]['res'] for idx in data['index']]
                data['log'] = [ans[idx]['log'] for idx in data['index']]
                dump(data, storage)

            data = load(storage)

            pred_value_in_cms = []
            for res in data["res"]:
                try:
                    pred_value_in_cms.append(self.parse_string(res))
                except ValueError:
                    pred_value_in_cms.append(0.)

            pred_value_in_cms = np.array(pred_value_in_cms) + 1e-8
        else:
            # regex parsing
            pred_value_in_cms = []
            n_errors_in_parsing = 0
            for pred in data["prediction"]:
                try:
                    parsed_value = self.parse_prediction(pred)
                except IndexError:
                    n_errors_in_parsing += 1
                    parsed_value = 1e-8

                pred_value_in_cms.append(parsed_value)

            print(f"Encounter {n_errors_in_parsing} errors in parsing")
            pred_value_in_cms = np.array(pred_value_in_cms) + 1e-8

        # Ground truth
        ground_truth_value_in_cms = []
        for answer in data["answer"]:
            value, unit = eval(answer)
            ground_truth_value_in_cms.append(value * self.get_multiplier(unit))
        ground_truth_value_in_cms = np.array(ground_truth_value_in_cms) + 1e-8

        # Calculate the score
        pred_gt = pred_value_in_cms / ground_truth_value_in_cms
        gt_pred = ground_truth_value_in_cms / pred_value_in_cms
        delta_2 = np.stack([pred_gt, gt_pred]).max(0) < 2.
        delta_1_point_5 = np.stack([pred_gt, gt_pred]).max(0) < 1.5

        data["eval_score_delta_2"] = delta_2
        data["eval_score_delta_1_point_5"] = delta_1_point_5

        final_score_dict = {
            "delta_2": delta_2.mean(),
            "delta_1_point_5": delta_1_point_5.mean()
        }
        for question_type in set(data["question_type"]):
            filtered_data = data[data["question_type"] == question_type]
            delta_2_per_question_type = filtered_data["eval_score_delta_2"].mean()
            delta_1_point_5_per_question_type = filtered_data["eval_score_delta_1_point_5"].mean()
            final_score_dict.update({f"{question_type}_delta_2": delta_2_per_question_type})
            final_score_dict.update({f"{question_type}_delta_1_point_5": delta_1_point_5_per_question_type})

        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict

```
python run.py --data MathVista_MINI --model llava_next_llama3 --verbose

```python
import json

import torch
import torch.distributed as dist

from vlmeval.config import supported_VLM #supported_VLM = {}
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import * #get_rank_and_world_size
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer

def build_model_from_config(cfg, model_name):
    import vlmeval.api
    import vlmeval.vlm
    config = cp.deepcopy(cfg[model_name])
    if config == {}:
        return supported_VLM[model_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.api, cls_name):
        return getattr(vlmeval.api, cls_name)(**config)
    elif hasattr(vlmeval.vlm, cls_name):
        return getattr(vlmeval.vlm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`')


def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')



def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `vlmeval/config.py` of check the output of the command \
        `vlmutil mlist all` in the terminal (you should first have vlmeval installed).
    To find all supported dataset names, please refer to the `vlmeval/dataset/__init__.py` file. The python script \
        to print all supported dataset names is as follows:
        ```python
        from vlmeval.dataset import SUPPORTED_DATASETS
        print(SUPPORTED_DATASETS)
        ```
        or you can check the output of the command `vlmutil dlist all` in the terminal.
    To find all supported video dataset default settings, please refer to the \
        `vlmeval/dataset/video_dataset_config.py` file.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
    {
        "model": {
            "GPT4o_20240806_T00_HIGH": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0,
                "img_detail": "high"
            },
            "GPT4o_20240806_T10_Low": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 1.0,
                "img_detail": "low"
            },
            "GPT4o_20241120": {}
        },
        "data": {
            "MME-RealWorld-Lite": {
                "class": "MMERealWorld",
                "dataset": "MME-RealWorld-Lite"
            },
            "MMBench_DEV_EN_V11": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN_V11"
            },
            "MMBench_Video_8frame_nopack": {},
            "Video-MME_16frame_subs": {
                "class": "VideoMME",
                "dataset": "Video-MME",
                "nframe": 16,
                "use_subtitle": true,
            }
        }
    }
    ```
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `vlmeval.vlm` or `vlmeval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_VLM` of `vlmeval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `vlmeval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.
    - Tip: The defined dataset in the `supported_video_datasets` of `vlmeval/dataset/video_dataset_config.py` \
        can be used as a shortcut.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API VLMs, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=bool, default=True, help='reuse auxiliary evaluation files')

    args = parser.parse_args()
    return args


def main():  #DATA,MODEL,CONFIG는 PARSE_ARG의 디폴트값 없음
    logger = get_logger('RUN')
    rank, world_size = get_rank_and_world_size()
    args = parse_args()
    use_config, cfg = False, None # if args.config is not None: True,load(args.config)지움
    assert len(args.data), '--data should be a list of data files'
#----------
def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size
#------------------`
    if rank == 0:
        if not args.reuse: #reuse 디폴트값 없음
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if not use_config: #use_config가 false시 작동
        for k, v in supported_VLM.items():
            #if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None: #args.retry 디폴트 none
            #    v.keywords['retry'] = args.retry
            #    supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        pred_root = osp.join(args.work_dir, model_name, eval_id)
        pred_root_meta = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

 

        for _, dataset_name in enumerate(args.data):
            if world_size > 1:
                dist.barrier()

            try:
                result_file_base = f'{model_name}_{dataset_name}.xlsx'
                #if use_config 지움 밑의 내용은 else:의 내용    
                dataset_kwargs = {}
                if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                    dataset_kwargs['model'] = model_name

                # If distributed, first build the dataset on the main process for doing preparation works
                if world_size > 1:
                    if rank == 0:
                        dataset = build_dataset(dataset_name, **dataset_kwargs)
                    dist.barrier()

                dataset = build_dataset(dataset_name, **dataset_kwargs)
                if dataset is None:
                    logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                    continue
#--------------------
def build_dataset(dataset_name, **kwargs):
    for cls in DATASET_CLASSES:
        if dataset_name in supported_video_datasets:
            return supported_video_datasets[dataset_name](**kwargs)
        elif dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)
#---------------------
 def supported_datasets(cls):
        return list(cls.DATASET_SETS)
#------------------
    warnings.warn(f'Dataset {dataset_name} is not officially supported. ')

    data_file = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    if not osp.exists(data_file):
        warnings.warn(f'Data file {data_file} does not exist. Dataset building failed. ')
        return None

    data = load(data_file)
    if 'question' not in [x.lower() for x in data.columns]:
        warnings.warn(f'Data file {data_file} does not have a `question` column. Dataset building failed. ')
        return None

    if 'A' in data and 'B' in data:
        if 'image' in data or 'image_path' in data:
            warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom MCQ dataset. ')
            return CustomMCQDataset(dataset=dataset_name, **kwargs)
        else:
            warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom Text MCQ dataset. ')
            return CustomTextMCQDataset(dataset=dataset_name, **kwargs)
    else:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom VQA dataset. ')
        return CustomVQADataset(dataset=dataset_name, **kwargs)
IMAGE_DATASET = [
    ImageCaptionDataset, ImageYORNDataset, ImageMCQDataset, ImageVQADataset, MathVision,
    MMMUDataset, OCRBench, MathVista, LLaVABench, MMVet, MTVQADataset, TableVQABench,
    MMLongBench, VCRDataset, MMDUDataset, DUDE, SlideVQA, MUIRDataset, CCOCRDataset,
    GMAIMMBenchDataset, MMERealWorld, HRBenchDataset, CRPE, MathVerse, NaturalBenchDataset,
    MIABench, OlympiadBench, WildVision, MMMath, QSpatial, Dynamath, MMGenBench, VizWiz, MMNIAH,
    CMMMU, VLRewardBench, WeMath, LogicVista, MMMUProDataset, CreationMMBenchDataset,
    ImageShortQADataset, MMAlignBench, OmniDocBench
]
DATASET_COLLECTION = [ConcatDataset, ConcatVideoDataset]
DATASET_CLASSES = IMAGE_DATASET + VIDEO_DATASET + TEXT_DATASET + CUSTOM_DATASET + DATASET_COLLECTIO
class CustomMCQDataset(ImageMCQDataset): #image_mcq.py class

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
class CustomVQADataset(ImageBaseDataset): #imagevqa.py에 잇는 클래스
    TYPE = 'VQA'

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError
#---------------------------
                # Handling Multi-Turn Dataset
                if dataset.TYPE == 'MT':
                    result_file_base = result_file_base.replace('.xlsx', '.tsv')

                result_file = osp.join(pred_root, result_file_base)
                
                # Reuse the previous prediction file if exists
                if rank == 0 and len(prev_pred_roots):
                    prev_result_files = []
                    prev_pkl_file_list = []
                    for root in prev_pred_roots[::-1]:
                        if osp.exists(osp.join(root, result_file_base)):
                            if args.reuse_aux: #디폴트 True
                                prev_result_files = fetch_aux_files(osp.join(root, result_file_base))
                            else:
                                prev_result_files = [osp.join(root, result_file_base)]
                            break
                        elif commit_id in root and len(ls(root)) and root != pred_root:
                            temp_files = ls(root, match=[dataset_name, '.pkl'])
                            if len(temp_files):
                                prev_pkl_file_list.extend(temp_files)
                                break
                    if not args.reuse:
                        prev_result_files = []
                        prev_pkl_file_list = []
                    if len(prev_result_files):
                        for prev_result_file in prev_result_files:
                            src = prev_result_file
                            tgt = osp.join(pred_root, osp.basename(src))
                            if not osp.exists(tgt):
                                shutil.copy(src, tgt)
                                logger.info(f'--reuse is set, will reuse the prediction file {src}.')
                            else:
                                logger.warning(f'File already exists: {tgt}')

                    elif len(prev_pkl_file_list):
                        for fname in prev_pkl_file_list:
                            target_path = osp.join(pred_root, osp.basename(fname))
                            if not osp.exists(target_path):
                                shutil.copy(fname, target_path)
                                logger.info(f'--reuse is set, will reuse the prediction pickle file {fname}.')
                            else:
                                logger.warning(f'File already exists: {target_path}')
#----------------------------
def fetch_aux_files(eval_file): 
    file_root = osp.dirname(eval_file)
    file_name = osp.basename(eval_file)

    eval_id = osp.basename(file_root)
    if eval_id[:3] == 'T20' and eval_id[9:11] == '_G':
        model_name = osp.basename(osp.dirname(file_root))
    else:
        model_name = eval_id
    
    dataset_name = osp.splitext(file_name)[0][len(model_name) + 1:]
    from vlmeval.dataset import SUPPORTED_DATASETS
    to_handle = []
    for d in SUPPORTED_DATASETS:
        if d.startswith(dataset_name) and d != dataset_name:
            to_handle.append(d)
    fs = ls(file_root, match=f'{model_name}_{dataset_name}')
    if len(to_handle):
        for d in to_handle:
            fs = [x for x in fs if d not in x]
    return fs
#---------------------------
                if world_size > 1:
                    dist.barrier()

                if model is None:
                    model = model_name  # which is only a name

                # Perform the Inference
                if dataset.MODALITY == 'VIDEO':
                    model = infer_data_job_video(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        result_file_name=result_file_base,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc)
                else: #mt 데이터 제거함
                    model = infer_data_job(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc, # default=4
                        ignore_failed=args.ignore)
# Ignore: will not rerun failed VLM inference parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
#--------------------------------- VLMEvalKit/vlmeval/inference.py
def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model, str) else model

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model
#---------------------------
                # Set the judge kwargs first before evaluation or dumping

                judge_kwargs = {
                    'nproc': args.api_nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}), #디폴트 none judge,judge_args 둘다
                }

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                if args.judge is not None:
                    judge_kwargs['model'] = args.judge
                else:
                    if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro']:
                        if listinstr(['WeMath'], dataset_name):
                            judge_kwargs['model'] = 'gpt-4o-mini'
                        else:
                            judge_kwargs['model'] = 'chatgpt-0125'
                    elif listinstr(['MMVet', 'LLaVABench', 'MMBench-Video'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4-turbo'
                    elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o-mini'
                    elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o'

                if rank == 0:
                    logger.info(judge_kwargs)

                if world_size > 1:
                    dist.barrier()

                # Only Rank 0 handles the evaluation part
                if rank == 0:
                    # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
                    if dataset_name in ['MMMU_TEST']:
                        result_json = MMMU_result_transfer(result_file)
                        logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                                    f'json file saved in {result_json}')
                        continue
                    elif 'MMT-Bench_ALL' in dataset_name:
                        submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                        logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                                    f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                                    f'submission file saved in {submission_file}')
                        continue

                    # Skip the evaluation part if only infer
                    if args.mode == 'infer': # 디폴트='all', choices=['all', 'infer']
                        continue

                    # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
                    if 'MLLMGuard_DS' in dataset_name:
                        logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
                        continue
                    elif 'AesBench_TEST' == dataset_name:
                        logger.info(f'The results are saved in {result_file}. '
                                    f'Please send it to the AesBench Team via huangyipo@hotmail.com.')
                        continue
                    elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                        logger.info(f'{dataset_name} is a test split without ground-truth. '
                                    'Thus only the inference part is supported for those datasets. ')
                        continue
                    elif dataset_name in [
                        'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                        'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
                    ] and not MMBenchOfficialServer(dataset_name):
                        logger.error(
                            f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')
                        continue

                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)
#-----------------
def proxy_set(s):
    import os
    for key in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
        os.environ[key] = s
#--------------------

                    # Perform the Evaluation
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
                    # Display Evaluation Results in Terminal
                    if eval_results is not None:
                        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                        logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                        logger.info('Evaluation Results:')
                        if isinstance(eval_results, dict):
                            logger.info('\n' + json.dumps(eval_results, indent=4))
                        elif isinstance(eval_results, pd.DataFrame):
                            if len(eval_results) < len(eval_results.columns):
                                eval_results = eval_results.T
                            logger.info('\n' + tabulate(eval_results))

                    # Restore the proxy
                    if eval_proxy is not None:
                        proxy_set(old_proxy)

                    # Create the symbolic links for the prediction files
                    files = os.listdir(pred_root)
                    files = [x for x in files if (f'{model_name}_{dataset_name}' in x or "status.json" in x)]
                    for f in files:
                        cwd = os.getcwd()
                        file_addr = osp.join(cwd, pred_root, f)
                        link_addr = osp.join(cwd, pred_root_meta, f)
                        if osp.exists(link_addr) or osp.islink(link_addr):
                            os.remove(link_addr)
                        os.symlink(file_addr, link_addr)

            except Exception as e:
                logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                                 'skipping this combination.')
                continue

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()
```
