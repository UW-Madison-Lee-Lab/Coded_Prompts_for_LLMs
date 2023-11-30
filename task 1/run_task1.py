import random
import numpy as np
import copy
from call_openai import call_gptapi

import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--integers', type=int, default=1, help='number of integers')
parser.add_argument('--samples', type=int, default=4, help='number of examples')
parser.add_argument('--apikey', type=str, default='use your openai key', help='openai api key')

parser.add_argument('--log_wandb', type=int, default=1,choices=[0,1])


args = parser.parse_args()

def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def generate_primes(v_min, v_max):
    """Generate a list of prime numbers smaller than max_v."""
    primes = []
    non_primes = []
    for i in np.arange(v_min, v_max):
        if is_prime(i):
            primes.append(i)
        else:
            non_primes.append(i)
    
    return primes, non_primes


    
def number2prompt(integers, prompt_type=None):
    #primes     = data['primes'    ]
    #non_primes = data['non_primes']
    #prompt = 'Given a sequence of numbers, output a binary string where 1 represents a prime number and 0 represents a non-prime number. For the sequence ['
    if prompt_type in ['multi', 'indiv1']:
        prompt = 'Please indicate whether the following statements are correct.\n'
        for i, integer in enumerate(integers[:-1]):
            prompt = prompt + '(' + str(i+1) + ') ' + str(integer) + ' is the largest prime number smaller than '+str(integers[-1])+'. \n'
            
        #if len(integers) == 0:
        #    prompt_cot = prompt#[:-2]
        #    prompt_sum = 'Provide 0 (for not prime) or 1 (for prime) for the single given number, with no commas, spaces, or additional text.'
        #else:
            #prompt_cot = prompt
            #prompt_sum = '], what would be the output? Directly output the binary string without any additional analysis.'
        prompt_sum = 'Provide a sequence of 0s (for wrong statement) and 1s (for correct statement) for the statements with no commas, spaces, or additional text.'
        prompt_cot = prompt + prompt_sum
    if prompt_type in ['indiv2']:
        prompt = ''
        for i, integer in enumerate(integers[:-1]):
            prompt = prompt + str(integer) + ' is the largest prime number smaller than '+str(integers[-1])+'.\n'
            
        #if len(integers) == 0:
        #    prompt_cot = prompt#[:-2]
        #    prompt_sum = 'Provide 0 (for not prime) or 1 (for prime) for the single given number, with no commas, spaces, or additional text.'
        #else:
            #prompt_cot = prompt
            #prompt_sum = '], what would be the output? Directly output the binary string without any additional analysis.'
        prompt_sum = 'Directly answer whether the above statement is true or false without any additional analysis.'
        prompt_cot = prompt + prompt_sum
    return {'cot':prompt_cot}

def map_pred2label(pred):
    if pred in ['1','true','True','true.','True.']:
        return '1'
    elif pred in ['0','false','False','false.','False.']:
        return '0'
    else:
        return 'XXX'
    

def mismatch_ratio(str1,str2,prompt_type=None):
    print('******', str1, str2)
    mis = 0
    cnt = 0
    
    F1 = {
        'p1l1': 0,
        'p1l0': 0,
        'p0l1': 0,
        'p0l0': 0,
    }
    
    if prompt_type in ['multi', 'indiv1']:
        for i in range(len(str1)):
            cnt += 1
            if str1[i] != map_pred2label(str2[i]):
                mis += 1
        
            if map_pred2label(str2[i]) == '1':
                if str1[i] == '1':
                    F1['p1l1'] += 1
                elif str1[i] == '0':
                    F1['p1l0'] += 1
            elif map_pred2label(str2[i]) == '0':
                if str1[i] == '1':
                    F1['p0l1'] += 1
                elif str1[i] == '0':
                    F1['p0l0'] += 1
        
    elif prompt_type in ['indiv2']:
        cnt += 1
        if str1 != map_pred2label(str2):
            mis += 1
        
        if map_pred2label(str2) == '1':
            if str1 == '1':
                F1['p1l1'] += 1
            elif str1 == '0':
                F1['p1l0'] += 1
        elif map_pred2label(str2) == '0':
            if str1 == '1':
                F1['p0l1'] += 1
            elif str1 == '0':
                F1['p0l0'] += 1
            
    return mis/cnt, F1



def get_llm_results(sorted_data, openai_key, model):
    sorted_labels = '0'*(len(sorted_data)-2)+'1'
    print(sorted_data)
    shuffled_array = np.random.permutation(np.arange(len(sorted_labels)))
    print(shuffled_array)
    labels = ''
    data = []
    for i in shuffled_array:
        labels += sorted_labels[i]
        data.append(sorted_data[i])
    data.append(sorted_data[-1])
    
    
    # multi
    prompt = number2prompt(data, prompt_type = 'multi')
    print(prompt['cot'])
    preds = call_gptapi(prompt['cot'], openai_key = openai_key, model=model)
    #labels = '0'*(len(data)-2)+'1'
    
    print(preds)
    print(labels)
    err_multi, F1_multi = mismatch_ratio(labels, preds, prompt_type = 'multi')
    
    # individual 1
    err_indiv = 0
    F1_indiv1 = {
        'p1l1': 0,
        'p1l0': 0,
        'p0l1': 0,
        'p0l0': 0,
    }
    for i in range(len(data)-1):
        sub_data = [data[i], data[-1]]
        prompt = number2prompt(sub_data, prompt_type = 'indiv1')
        if i == 0:
            print(prompt['cot'])
        preds = call_gptapi(prompt['cot'], openai_key = openai_key, model=model)
        if i != len(data)-2:
            labels = '0'
        else:
            labels = '1'
        if i != -1:
            print(preds)
            print(labels)
        print('*******************', preds)
        print('*******************', labels)
        err, F1 = mismatch_ratio(labels, preds, prompt_type = 'indiv1')
        err_indiv += err
        for key in F1_indiv1.keys():
            F1_indiv1[key] += F1[key]
        print('*******************', mismatch_ratio(labels, preds, prompt_type = 'indiv1'))
    err_indiv1 = err_indiv/(len(data)-1)
    
    # individual 2
    err_indiv = 0
    F1_indiv2 = {
        'p1l1': 0,
        'p1l0': 0,
        'p0l1': 0,
        'p0l0': 0,
    }
    for i in range(len(data)-1):
        sub_data = [data[i], data[-1]]
        prompt = number2prompt(sub_data, prompt_type = 'indiv2')
        if i == 0:
            print(prompt['cot'])
        preds = call_gptapi(prompt['cot'], openai_key = openai_key, model=model)
        if i != len(data)-2:
            labels = '0'
        else:
            labels = '1'
        if i != -1:
            print(preds)
            print(labels)
        print('*******************', preds)
        print('*******************', labels)
        err, F1 = mismatch_ratio(labels, preds, prompt_type = 'indiv2')
        err_indiv += err
        for key in F1_indiv2.keys():
            F1_indiv2[key] += F1[key]
        print('*******************', mismatch_ratio(labels, preds, prompt_type = 'indiv2'))
    err_indiv2 = err_indiv/(len(data)-1)
    

    return err_multi, err_indiv1, err_indiv2, F1_multi, F1_indiv1, F1_indiv2

def F1_score(F1):
    delta = 0.0001
    precision = F1['p1l1']/(F1['p1l1']+F1['p1l0']+delta)
    recall    = F1['p1l1']/(F1['p1l1']+F1['p0l1']+delta)
    return 2*precision*recall/(precision+recall)

if 1: #test basic accuracy:
    k = args.samples
    base = args.integers
    primes = generate_primes(v_min=10**base, v_max=10**(base+1))[0]
    EP = 400
    indexes = np.arange(len(primes))[k+1:]
    s_err_multi, s_err_indiv1, s_err_indiv2 = 0, 0, 0
    count = 0
    
    F1_multi = {
        'p1l1': 0,
        'p1l0': 0,
        'p0l1': 0,
        'p0l0': 0,
    }
    F1_indiv1 = {
        'p1l1': 0,
        'p1l0': 0,
        'p0l1': 0,
        'p0l0': 0,
    }
    F1_indiv2 = {
        'p1l1': 0,
        'p1l0': 0,
        'p0l1': 0,
        'p0l0': 0,
    }
    for ep in np.arange(1, EP+1):
        print('EP = ', ep)
        end_index = random.choice(indexes)
        integers = primes[end_index-k-1:end_index]
        
        #print(integers)
        err_multi, err_indiv1, err_indiv2, F1_multi_, F1_indiv1_, F1_indiv2_ = get_llm_results(integers, openai_key = args.apikey, model = 'gpt-4')
        print(err_multi, err_indiv1, err_indiv2)
        for key in F1_multi.keys():
            F1_multi [key] += F1_multi_ [key]
        for key in F1_indiv1.keys():
            F1_indiv1[key] += F1_indiv1_[key]
        for key in F1_indiv2.keys():
            F1_indiv2[key] += F1_indiv2_[key]
            
        count += 1
        s_err_multi += err_multi
        s_err_indiv1 += err_indiv1
        s_err_indiv2 += err_indiv2
        print(1-s_err_multi/count, 1-s_err_indiv1/count, 1-s_err_indiv2/count)
    print('F1: ', F1_score(F1_multi), F1_score(F1_indiv1), F1_score(F1_indiv2))


