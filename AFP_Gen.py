#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:37:49 2022

@author: kevin kim (kekim@hmc.edu)
"""

import tensorflow as tf
import numpy as np
import numpy
from tensorflow.python.ops import bitwise_ops
from tensorflow import keras
from tensorflow.keras import layers
import random

import sys

  # This is the standard AFP rounding with positive and unused bits
  # Need to implement stochastic rounding for rounding
  # use 1 random value per block?
  # Can use the exponent as random seed?
  # break up 32 float into individual components
  
  
  

def tf_custom_round(t_value, round_mode = "truncate", skip_processing = False, block_round_mode = "fine_grain_2_contiguous_exponent_reuse",
                 num_exp = 8, num_mantissa = 23, radix_exp = 2, block_size = 16, 
                 radix_mantissa = 2, is_tensor = True):
    """
    block_size: number of elements in a block (default)
    t_value: 
    """
    
    #NOTE: Shared field should be in the beginning
    #NOTE: Exponent, then positive (total 8 bits of information)
    
    #GENERATE MASKS
    MAN_MASK = 0x007FFFFF 
    EXP_MASK = 0x7F800000
    SIGN_MASK = 0x80000000
    SIGN_EXP_MASK = 0xFF800000
    GUARD_MASK = 0x00000007
    GUARD_RIGHT = 0x00000001
    GUARD_MID = 0x00000002 
    GUARD_LEFT = 0x00000004
    
    # This mask is shifted to the right by reduce_num
    # Upper bits ORd with 1's
    TRUNCATE_MASK = 0xFF800000
    
    # This will point to the digit after the last digit to keep
    NEAREST_PTR = 0x00400000
    LOWEST_PTR = 0x00000001
    
    MAN_OVERFLOW = 0x00800000
    
    LARGEST_DENORMAL_EXP_MASK = 0x7F800000
    SMALLEST_DENORMAL_EXP_MASK = 0x00000000
    
    IMPLICIT_ONE = 0x00800000
    
    # Number of bits to reduce based on new number of mantissa bits
    reduce_num = 23 - num_mantissa
    
    # need to traverse the tensorfine_grain
    # convert tensor to numpy array
    # if skip_processing == True:
    #   length = 1
    # else:
    #   length = len(t_value)
    
    
    # if (skip_processing == False):
    #   if is_tensor:
    #     x = t_value[t_num].numpy()
    #   else:
    #     x = t_value[t_num]
    # else:
    #     x = t_value
    
    # temp_convert_x = tf.bitcast(x, tf.int32)
    # temp_exp_bits = bitwise_ops.bitwise_and(temp_convert_x, EXP_MASK)
    # temp_exp_bits = bitwise_ops.right_shift(temp_exp_bits,23)
    # temp_max = tf.math.reduce_max(temp_exp_bits)
    # temp_min = tf.math.reduce_min(temp_exp_bits)
    # temp_range = temp_max - temp_mint
    #print("local exp range: ", temp_range)
    
    #print("#############################")
    #print("This is the original tensor: ", x)
    x = t_value 
    
    num_mantissa_save = num_mantissa
    convert_x = tf.bitcast(x, tf.int32)
    
    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)
    
    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)
    
    # look to see if we need deepcopy here
    saved_exp_bits = exp_bits
    saved_man_bits = man_bits
    
    diff_tensor = tf.math.subtract(max_exp,exp_bits)
    
    num_mantissa = tf.where(tf.equal(diff_tensor,0), num_mantissa_save - 2, num_mantissa)
    num_mantissa = tf.where(tf.equal(diff_tensor,1), num_mantissa_save - 1, num_mantissa)
    num_mantissa = tf.where(tf.equal(diff_tensor,2), num_mantissa_save , num_mantissa)
    num_mantissa = tf.where(tf.equal(diff_tensor,3), num_mantissa_save + 1, num_mantissa)
    num_mantissa = tf.where(tf.equal(diff_tensor,4), num_mantissa_save + 2, num_mantissa)
    num_mantissa = tf.where(tf.equal(diff_tensor,5), num_mantissa_save + 3, num_mantissa)
    num_mantissa = tf.where(tf.equal(diff_tensor,6), num_mantissa_save + 4, num_mantissa)
    num_mantissa = tf.where(tf.greater_equal(diff_tensor,7), num_mantissa_save + 4, num_mantissa)
    
    num_mantissa_save2 = num_mantissa
    # num_mantissa = tf.where(tf.equal(diff_tensor,0), num_mantissa -1, num_mantissa)
    # # num_mantissa = tf.where(tf.equal(diff_tensor,1), num_mantissa , num_mantissa)
    # num_mantissa = tf.where(tf.equal(diff_tensor,2), num_mantissa + 1 , num_mantissa)
    # num_mantissa = tf.where(tf.greater_equal(diff_tensor,3), num_mantissa + 1, num_mantissa)
    ##### Same Sign Code
    
    # truncate_to_zero = tf.math.greater(diff_tensor, num_mantissa)
    # sum_of_sign = tf.math.reduce_sum(sign_bit, axis=1, keepdims=False)
    bool_sign_bit = tf.cast(sign_bit,tf.bool)
    
    ######
    
    total_elements =  tf.size(bool_sign_bit)
    
    #NOTE: Do we need to divide by two?
    num_blocks = total_elements // (block_size // 2)
    
    flattened_sign = tf.reshape(bool_sign_bit, [num_blocks, (block_size // 2)])
    
    # alternative_neg_sign = tf.where(tf.equal(t_value,0.0), 1, 0)
    # alternative_neg_sign = tf.where(bool_sign_bit, 1, alternative_neg_sign)  
    alternative_pos_sign = tf.where(flattened_sign, 1, 0)
    #alternative_pos_sign = tf.where(tf.greater(diff_tensor, num_mantissa), 0, alternative_pos_sign)
    
    sum_of_pos_sign = tf.math.reduce_sum(alternative_pos_sign, axis=1, keepdims=False)
    # sum_of_neg_sign = tf.math.reduce_sum(alternative_neg_sign, axis=1, keepdims=False)
    
    # all_one_sign = tf.math.logical_or(tf.math.equal(sum_of_pos_sign, 0),
    #                                   tf.math.equal(sum_of_neg_sign, block_size))
    
    all_one_sign = tf.math.equal(sum_of_pos_sign, 0)
    
    #num_mantissa_block = tf.where(all_one_sign, num_mantissa + 1, num_mantissa)
    
    all_one_sign = tf.repeat(all_one_sign, flattened_sign.shape[1] )
    all_one_sign = tf.reshape(all_one_sign, bool_sign_bit.shape)
    
    # Use sign bit for each 8 bit value
    
    # second_window = tf.greater_equal(diff_tensor, num_mantissa)
    #all_one_sign = tf.logical_and(all_one_sign, second_window)
    #num_mantissa = tf.where(all_one_sign, num_mantissa + num_mantissa, num_mantissa)    
    num_mantissa = tf.where(all_one_sign, num_mantissa + 1, num_mantissa)
    
    num_mantissa_save = num_mantissa
    # print("sign num_mantissa: ", num_mantissa)
    #################################################
    # Find free encodings in lower half - exponent is < max_exp
    
    # extra_bits = num_mantissa - 1 - unused_imp
    extra_bits = 0
    ##############################
    # Find free intervals in the upper half - All these values have max exp
    
    
    used2_imp1 = bitwise_ops.bitwise_and(saved_man_bits, 0x00400000)
    used2_imp2 = bitwise_ops.bitwise_and(saved_man_bits, 0x00400000)
    # used2_imp3 = bitwise_ops.bitwise_and(saved_man_bits, 0x00400000)
    # used2_imp4 = bitwise_ops.bitwise_and(saved_man_bits, 0x00400000)
    # used2_imp5 = bitwise_ops.bitwise_and(saved_man_bits, 0x00040000)
    
    # check if value has max_exponent
    max_exp_sel1 = tf.math.equal(diff_tensor, 0)
    max_exp_sel2 = tf.math.equal(diff_tensor, 1)
    # max_exp_sel3 = tf.math.equal(diff_tensor, 2)        
    # max_exp_sel4 = tf.math.equal(diff_tensor, 3)
    
    # check 110, 111, 101 - can represent 2 extra bits
    # check 1001, 1111 - 
    # used2_imp1 = tf.equal(used2_imp1, (int)(0x00400000))
    # used2_imp2 = tf.equal(used2_imp2, (int)(0x00200000))
    # used2_imp3 = tf.equal(used2_imp3, (int)(0x00100000)) 
    # used2_imp4 = tf.equal(used2_imp4, (int)(0x00080000))
    # used2_imp5 = tf.equal(used2_imp5, (int)(0x00040000))
    
    used2_imp1 = tf.cast(used2_imp1, tf.bool)
    used2_imp2 = tf.cast(used2_imp2, tf.bool)
    # used2_imp3 = tf.cast(used2_imp3, tf.bool)
    # used2_imp4 = tf.cast(used2_imp4, tf.bool)
    # used2_imp5 = tf.cast(used2_imp5, tf.bool)
    
    used2_imp1 = tf.logical_and(used2_imp1, max_exp_sel1)
    used2_imp2 = tf.logical_and(used2_imp2, max_exp_sel2)
    # used2_imp3 = tf.logical_and(used2_imp3, max_exp_sel3)
    # used2_imp4 = tf.logical_and(used2_imp4, max_exp_sel4)
    # used2_imp5 = tf.logical_and(used2_imp5, max_exp_sel)
    
    #########################
    # split into 2 halves checking for upper 2 bits
    
    total_elements =  tf.size(used2_imp1)
    num_blocks = total_elements // (block_size // 2)
    
    flattened_used2_imp1 = tf.reshape(used2_imp1, [num_blocks, (block_size // 2)])
    flattened_used2_imp2 = tf.reshape(used2_imp2, [num_blocks, (block_size // 2)])
    
    #########################
    
    used2_imp1 = tf.reduce_any(flattened_used2_imp1, axis=1)
    used2_imp2 = tf.reduce_any(flattened_used2_imp2, axis=1)
    # used2_imp1 = tf.reduce_any(used2_imp1, axis=1)
    # used2_imp2 = tf.reduce_any(used2_imp2, axis=1)    
    # used2_imp3 = tf.reduce_any(used2_imp3, axis=1)
    # used2_imp4 = tf.reduce_any(used2_imp4, axis=1)
    # used2_imp5 = tf.reduce_any(used2_imp5, axis=1)
    
    unused2_imp1 = tf.logical_not(used2_imp1)
    unused2_imp2 = tf.logical_not(used2_imp2)    
    # unused2_imp3 = tf.logical_not(used2_imp3)
    # unused2_imp4 = tf.logical_not(used2_imp4)
    # unused2_imp5 = tf.logical_not(used2_imp5)
    
    
    # unused2_imp1 = tf.logical_and(unused2_imp1,  tf.greater_equal(num_mantissa_block,2))
    # unused2_imp2 = tf.logical_and(unused2_imp2,  tf.greater_equal(num_mantissa_block,3))  
    # unused2_imp3 = tf.logical_and(unused2_imp3,  tf.greater_equal(num_mantissa_block,4))
    # unused2_imp4 = tf.logical_and(unused2_imp4,  tf.greater_equal(num_mantissa_block,5))
    # unused2_imp5 = tf.logical_and(unused2_imp5,  tf.greater_equal(num_mantissa_block,6))
    
    unused2_imp1 = tf.repeat(unused2_imp1, flattened_used2_imp1.shape[1])
    unused2_imp2 = tf.repeat(unused2_imp2, flattened_used2_imp1.shape[1])
    # unused2_imp3 = tf.repeat(unused2_imp3, man_bits.shape[1])
    # unused2_imp4 = tf.repeat(unused2_imp4, man_bits.shape[1])
    # unused2_imp5 = tf.repeat(unused2_imp5, man_bits.shape[1])
    
    unused2_imp1 = tf.reshape(unused2_imp1, man_bits.shape)
    unused2_imp2 = tf.reshape(unused2_imp2, man_bits.shape)
    # unused2_imp3 = tf.reshape(unused2_imp3, man_bits.shape)
    # unused2_imp4 = tf.reshape(unused2_imp4, man_bits.shape)
    # unused2_imp5 = tf.reshape(unused2_imp5, man_bits.shape)  
      
    #####################
    
    # unused2_imp1 = tf.logical_and(unused2_imp1,  tf.greater_equal(num_mantissa,2+2))
    # unused2_imp2 = tf.logical_and(unused2_imp2,  tf.greater_equal(num_mantissa,3+2))  
    # unused2_imp3 = tf.logical_and(unused2_imp3,  tf.greater_equal(num_mantissa,4+2))
    # unused2_imp4 = tf.logical_and(unused2_imp4,  tf.greater_equal(num_mantissa,5+2))
    # unused2_imp5 = tf.logical_and(unused2_imp5,  tf.greater_equal(num_mantissa,6+2))
    
    unused2_imp1 = tf.logical_and(unused2_imp1, max_exp_sel1)
    unused2_imp2 = tf.logical_and(unused2_imp2, max_exp_sel2)
    # unused2_imp3 = tf.logical_and(unused2_imp3, max_exp_sel3)
    # unused2_imp4 = tf.logical_and(unused2_imp4, max_exp_sel4)
    
    unused2_imp1 = tf.logical_and(unused2_imp1, tf.logical_not(all_one_sign))
    unused2_imp2 = tf.logical_and(unused2_imp2, tf.logical_not(all_one_sign))
    
    
    num_mantissa = tf.where(tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)), 
                            num_mantissa + 1, num_mantissa)
    
    num_mantissa = tf.where(tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)), 
                            num_mantissa + 1, num_mantissa)
    
    # shift_to_zero is true for values that would be truncated to zero
    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 
    
    # shift_to_zero_round_up is true for values with 1 just to the right of mantissa bits
    # shift_to_zero_round_up = tf.logical_and(tf.math.greater_equal(diff_tensor, num_mantissa),
    #                                         tf.math.less(diff_tensor, num_mantissa + 1))
    shift_to_zero_round_up = tf.math.equal(diff_tensor, num_mantissa)
    
    # Exponent only representation for values that would truncate to zero
    twos_tensor = tf.ones_like(man_bits)
    twos_tensor = tf.add(twos_tensor, twos_tensor)
    extra_bits = tf.where(tf.less(extra_bits, 0), 0, extra_bits)
    powers_of_two = tf.math.pow(twos_tensor, extra_bits)
    powers_of_two = tf.where(tf.less_equal(powers_of_two,1), 0, powers_of_two)
    
    # shift_to_zero_round_up_exp indicates numbers that are within the range of exponent only format
    shift_to_zero_round_up_exp = tf.logical_and(tf.math.greater_equal(diff_tensor, 
                                                                      num_mantissa),
                                            tf.math.less(diff_tensor, num_mantissa + powers_of_two)) 
    
    # update this to add 1 bit such that a 1 in the guard bit will be rounded up
    
    # shifting exponent bits back to original position
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    
    # Shifting mantissa bits diff_tensor bits to the right
    # Simulate block based representation
    man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)
    
    # Need to mask out the bits lower than the num_mantissa bits
    #mask = int(TRUNCATE_MASK)
    mask = tf.ones_like(man_bits)
    mask = tf.where(True, tf.cast(TRUNCATE_MASK, tf.int32), TRUNCATE_MASK)
    num_mantissa_minus_1 = tf.cast(tf.subtract(num_mantissa, 1), tf.int32)
    
    # Using arithmetic right shift to add 1's on the left side of the mask
    mask = tf.cast(mask, tf.int32)
    mask = tf.bitwise.right_shift(mask, num_mantissa_minus_1)
    #mask = tf.bitcast(mask, tf.int32)
    
    # guard mask for rounding
    
    guard_mask = NEAREST_PTR
    guard_mask = tf.repeat(guard_mask, tf.size(sign_bit))
    guard_mask = tf.reshape(guard_mask, sign_bit.shape)
    guard_mask = tf.bitwise.right_shift(guard_mask, tf.cast(num_mantissa - 1, tf.int32))
    guard_mask = tf.bitcast(guard_mask, tf.int32)    
    # Nearest Rounding code
    # guard_mask = int(NEAREST_PTR)
    # for j in range(0, num_mantissa - 1):
    #   guard_mask = guard_mask >> 1    
    
    # guard_mask = bitwise_ops.right_shift(guard_mask, reduce_num -1)
    
    # Need to shift guard_mask to the left when exponent is smaller than max_exp
    guard_mask = bitwise_ops.left_shift(guard_mask, diff_tensor)
    half_one = bitwise_ops.bitwise_and(convert_x, guard_mask)    
    round_value = tf.less(diff_tensor, num_mantissa)
    zeros_tensor = tf.zeros_like(exp_bits)
    
    # make sure value is within mantissa window before rounding
    half_one= tf.where(round_value, half_one, zeros_tensor) 
    half_bool = tf.cast(half_one, tf.bool)
    
    # build value to add for rounding 
    # exponent for round value of max_exp values
    guard_exp_bits = tf.subtract(saved_exp_bits, num_mantissa -1)
    # need to add to exponent of round value for smaller values
    guard_exp_bits = tf.add(guard_exp_bits, diff_tensor)
    guard_exp_bits = bitwise_ops.left_shift(guard_exp_bits, 23)
    round_up = bitwise_ops.bitwise_or(guard_exp_bits, sign_bit)
    ones_tensor = tf.ones_like(exp_bits)
    
    # y is the round value
    y = tf.bitcast(round_up, tf.float32)
    
    # building final FP32 value
    saved_man_bits = man_bits
    
    man_bits = bitwise_ops.bitwise_and(man_bits, mask)
    man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)
    final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    final_value = bitwise_ops.bitwise_or(final_value, man_bits)
    
    zeros_tensor = tf.zeros_like(man_bits)
    # round_up_value = tf.add(saved_exp_bits , 1)
    # round value for numbers with 1 to the right of the mantissa window
    round_up_value = max_exp - (num_mantissa - 1)
    round_up_value = bitwise_ops.left_shift(round_up_value, 23)
    round_up_value = bitwise_ops.bitwise_or(round_up_value, sign_bit)
    
    # round value for numbers shifted to exponent only window
    round_exp_only = bitwise_ops.bitwise_and(saved_man_bits, 0x00700000)
    round_exp_only = tf.cast(round_exp_only, tf.bool)
    exp_only_value = tf.where(round_exp_only, tf.add(saved_exp_bits, 1), saved_exp_bits)
    exp_only_value = bitwise_ops.left_shift(exp_only_value, 23)
    #exp_only_value = bitwise_ops.bitwise_or(exp_only_value, sign_bit)
    
    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)
    final_value = tf.where(shift_to_zero_round_up, round_up_value, final_value)
    final_value = tf.where(shift_to_zero_round_up_exp, exp_only_value, final_value)
    
    # Final check to make sure we are only using bits we have
    final_value = bitwise_ops.bitwise_and(final_value, mask)
    
    x_temp = tf.bitcast(final_value, tf.float32)
    
    # x = tf.where(half_bool, x + y, x)
    
    #####################
    # Stochastic MSFP
    #####################
    
    # remain_mask should be the inverted version of the regular bit mask
    # reduce num, remain_mask, and random_prob need to be tensors
    # TODO: for block floating point, need to take care of case where
    # entire value is truncated. Implicit 1 needs to be factored in for the 
    # probability calculation also.
    
    # need to add implicit 1
    
    # Stochastic AFP
    # remain_mask should be the inverted version of the regular bit mask
    # reduce num, remain_mask, and random_prob need to be tensors
    # This subtraction gives us 0 or 1 - for all positive and unused bits
    
    # need to create a new mask (not man_mask) similar to msfp_stochastic
    # extra_bit = num_mantissa - num_mantissa_save2
    
    # actual number of mantissa bits
    # actual_mantissa = extra_bit + num_mantissa_save - 2
    actual_mantissa = num_mantissa - diff_tensor
    
    less_than_7 =  tf.less(diff_tensor,7) 
    equal_7 = tf.equal(diff_tensor,7)
    greater_than_7 = tf.greater(diff_tensor,7)
    # if diff is < 7, do not shift, invert mask and use actual mantissa to figure out 
    # round bit value and guard bit value (round bit is max - actual_mantissa + 1)
    # guard bit is max-actual_mantissa
    
    # if diff  == 7. then round bit is (max - actual_mantissa +2)
    # guard is (max-actual_mantissa+1)
    # if diff > 7, need to add implicit 1, then shift to the right 
    
    # diff  < 7 case
    new_mask = tf.cast(TRUNCATE_MASK, tf.int32)
    new_shift = actual_mantissa - 1
    # diff == 7 case
    new_shift = tf.where(equal_7, new_shift - 1, new_shift)
    # diff > 7
    new_diff_tensor = tf.where(tf.greater(diff_tensor,num_mantissa-1), num_mantissa -7 -1, num_mantissa - diff_tensor  )
    new_shift = tf.where(greater_than_7, new_diff_tensor, new_shift )
    
    # Using arithmetic right shift to add 1's on the left side of the mask
    new_mask = tf.bitwise.right_shift(new_mask, new_shift)
    # new_mask_random = tf.bitwise.right_shift(tf.cast(TRUNCATE_MASK, tf.int32), new_shift - 1)
    
    remain_mask = bitwise_ops.invert(new_mask)
    # print("remain_mask: ", remain_mask)
    # random_mask = bitwise_ops.invert(new_mask_random)
    # print("random_mask: ", random_mask)
    
    new_man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)
    new_man_bits = tf.where(tf.equal(exp_bits, 0), new_man_bits, bitwise_ops.bitwise_or(new_man_bits, 0x00800000))
    new_man_bits_greater_7 = bitwise_ops.right_shift(new_man_bits,diff_tensor-7)
    new_man_bits = tf.where(greater_than_7, new_man_bits_greater_7, new_man_bits)
    remainder_value = bitwise_ops.bitwise_and(new_man_bits, remain_mask)
    
    # reduce_num = 23
    # final_diff = actual_mantissa - 1
    # # diff < 7
    # reduce_num = tf.subtract(reduce_num, final_diff)
    # # diff >= 7
    # reduce_num = tf.where(tf.logical_or(equal_7, greater_than_7), 23 - (num_mantissa - 7 - 1),reduce_num)
    
    
    # base = 2
    # epsilon_int = tf.math.pow(base, reduce_num)
    # epsilon_int = tf.cast(epsilon_int, tf.int32)
    
    # New epsilon_int code
    epsilon_int = remain_mask + 1
    
    guard_mask = bitwise_ops.right_shift(epsilon_int, 1)
    guard_bit = bitwise_ops.bitwise_and(guard_mask, new_man_bits)
    final_round = tf.cast(guard_bit, tf.bool)
    # epsilon_int = epsilon_int + 1
    # epsilon_int_max = tf.reduce_max(epsilon_int)
    # epsilon_int_max = tf.math.pow(2,23)
    # epsilon_int_max += 1
    
    # probability = tf.cast(tf.divide(remainder_value, epsilon_int), tf.float32)
    
    # random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)
    
    # # remainder_value = tf.where(tf.equal(x, 0.0), 0, remainder_value)
    
    # # random_prob_int = tf.add(random_int, remainder_value)
    
    # final_probability = tf.add(random_prob, probability)
    # ones_tensor = tf.ones_like(random_prob)
    # final_round = tf.math.greater_equal(final_probability, ones_tensor)
    
    
    # print("new_shift: ", new_shift)
    # print("max_exp: ", max_exp -127)
    # print("diff: ", diff_tensor)
    # print("all_one_sign: ", all_one_sign)
    # print("unused 0: ", tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)))
    # print("unused 1: ", tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)))
    # print("num_mantissa: ", num_mantissa)
    # print("reduce_num: ", reduce_num)
    # print("epsilon_int: ", epsilon_int)
    # print("guard_mask: ", guard_mask)
    # print("guard_bit: ", guard_bit)
    # print("final_round: ", final_round)
    # print("remainder_value: ", remainder_value)
    # print("probability: ", probability)
    # print("random_prob: ", random_prob)
    # print("final_prob:", final_probability)
    # print("final_round: ", final_round)
    
    
    x_temp = tf.where(final_round, x_temp + y, x_temp)
    x_temp = tf.where(tf.equal(max_exp,0), x, x_temp)
    # print("x+y or x (rounded): ", x_temp )
    # print("x: ", x)
    # print("y: ", y)
    #new_round_exp = tf.subtract(max_exp - 1, actual_mantissa)
    ###########
    
    ##########################
    # Denormal values
    ##########################
    
    # exp_bits = bitwise_ops.bitwise_and(convert_x,EXP_MASK)
    # # detect denormal values by looking for 1 in exponent
    # zeroes_tensor = tf.zeros_like(exp_bits)
    # denormal = tf.math.equal(exp_bits, zeroes_tensor)
    
    # x = tf.where(denormal, x, final_value) 
        
    
    t_value = x_temp
    
    return t_value


# %%

#Create 16 element vector with numbers [0,15]
t_value = np.arange(0,16)
t_value = tf.convert_to_tensor(t_value, dtype = tf.float32) 

tf.print(t_value)

tf_custom_round(t_value, round_mode = "truncate", skip_processing = False, block_round_mode = "fine_grain_2_contiguous_exponent_reuse",num_exp = 8, num_mantissa = 23, radix_exp = 2, block_size = 128, radix_mantissa = 2, is_tensor = True)



