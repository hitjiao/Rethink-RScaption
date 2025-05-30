import openai
import json
import argparse
import time
import re

def filter_numbers(text):
    # 使用正则表达式提取数字
    return re.sub(r'[^1-5]', '', text)
def filter_numbers2(text):
    # 使用正则表达式提取数字
    numbers = re.findall(r'\d', text)
    # 确保返回的数字长度为 5
    if len(numbers) == 5:
        return ''.join(numbers[:5])
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, default=r'F:\pythonProject\geval-main\prompts\summeval\richness_rank.txt')
    argparser.add_argument('--save_fp', type=str, default=r'F:\pythonProject\geval-main\results\4o\40_fin_rich.json')
    argparser.add_argument('--candidate1', type=str, default=r'F:\pythonProject\geval-main\my_result\Sy_4.14_.json')
    argparser.add_argument('--candidate2', type=str, default=r'F:\pythonProject\geval-main\others\geochat\geochat-SydneyCaptions.json')
    argparser.add_argument('--candidate3', type=str, default=r'F:\pythonProject\geval-main\others\minigpt\results_sy.json')
    #argparser.add_argument('--candidate4', type=str, default=r'F:\pythonProject\geval-main\others\qwen\qwen-Sy.json')
    argparser.add_argument('--candidate4', type=str, default=r'F:\pythonProject\geval-main\others\RS-llava\RS-llava-SydneyCaptions.json')
    argparser.add_argument('--candidate5', type=str, default=r'F:\pythonProject\geval-main\others\VHM\VHM_Sydney.json')
    argparser.add_argument('--key', type=str, default='sk-jMFeZdioK7urgkCMwZgo8P76ifj3VuH3UeoJ0g46TLDw3Hqd')
    argparser.add_argument('--base', type=str, default='https://api.nuwaapi.com/v1')
    argparser.add_argument('--model', type=str, default='gpt-4o')
    args = argparser.parse_args()
    openai.api_key = args.key
    openai.api_base = args.base

    new_json = []
    ct, ignore = 0, 0

    with open(args.candidate1, "r", encoding="utf-8") as file1:
        candidates1_json = json.load(file1)
    with open(args.candidate2, "r", encoding="utf-8") as file2:
        candidates2_json = json.load(file2)
    with open(args.candidate3, "r", encoding="utf-8") as file3:
        candidates3_json = json.load(file3)
    with open(args.candidate4, "r", encoding="utf-8") as file4:
        candidates4_json = json.load(file4)
    with open(args.candidate5, "r", encoding="utf-8") as file5:
        candidates5_json = json.load(file5)
    with open(args.prompt_fp, "r", encoding="utf-8") as file:
        prompt = file.read()

    for key, value in candidates1_json.items():
        if int(key)<10000:
            candidate1 = value[0]
            candidate2 = candidates2_json[key][0]
            candidate3 = candidates3_json[key][0]
            candidate4 = candidates4_json[key][0]
            candidate5 = candidates5_json[key][0]

            cur_prompt = prompt.replace('{{candidate1}}', candidate1).replace('{{candidate2}}', candidate2).replace('{{candidate3}}', candidate3).replace('{{candidate4}}', candidate4).replace('{{candidate5}}', candidate5)
            # print(cur_prompt)

            while True:
                try:
                    _response = openai.ChatCompletion.create(
                        model=args.model,
                        messages=[{"role": "system", "content": cur_prompt}],
                        temperature=2,
                        max_tokens=5,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        n=10
                    )
                    print(key)
                    time.sleep(0.5)

                    # 过滤非数字字符
                    all_responses = [filter_numbers2(filter_numbers(_response['choices'][i]['message']['content'])) for i in
                                     range(len(_response['choices']))]
                    new_json.append(all_responses)
                    ct += 1
                    break
                except Exception as e:
                    print(e)
                    if "limit" in str(e):
                        time.sleep(2)
                    else:
                        ignore += 1
                        print('ignored', ignore)
                        break
    print('ignored total', ignore)
    with open(args.save_fp, 'w', encoding="utf-8") as f:
        json.dump(new_json, f, indent=4, ensure_ascii=False)
