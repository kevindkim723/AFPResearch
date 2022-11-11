import tensorflow as tf
import numpy as np
import numpy
from tensorflow.python.ops import bitwise_ops
from tensorflow import keras
from tensorflow.keras import layers
import random

import sys

# import tensorflow_probability as tfp

# import rounding

def tf_custom_round(t_value, round_mode = "truncate", skip_processing = False, block_round_mode = "fine_grain_2_contiguous_exponent_reuse",
                 num_exp = 8, num_mantissa = 23, radix_exp = 2, block_size = 128, 
                 radix_mantissa = 2, is_tensor = True):
  '''
    is_tensor will toggle between treating t_value as a tensor vs a non-tensor
    Assume Float 32 - TODO: check data type and round accordingly
    Round all values inside t_value tensor based on the following:
    round_mode: truncate, nearest, stochastic, hw_stochastic, custom
    num_exp: number of bits for exponent
    num_mantissa: number of bits for mantissamantissa_
    radix_exp: base number for representation of exponent
    radix_mantissa: base number for representation of mantissa 
  '''
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

  if (round_mode == "truncate"):
    print("rounding with truncate mode")
    # Code to truncate the mantissa to the new number of bits
    mask = int(TRUNCATE_MASK)
    ##print("This is the original mask: ",hex(mask))tf
    for j in range(0, num_mantissa):
      mask = mask >> 1
      mask = mask | 0x80000000
    #mask = mask >> reduce_num
    #print("This is the new mask: ",hex(mask))
    # Need to cast float into integer format in order to apply bitwise operations
    # then convert back
    convert_x = tf.bitcast(x, tf.int32)
    convert_x = tf.bitwise.bitwise_and(convert_x, mask)
    x = tf.bitcast(convert_x, tf.float32)

    t_value = x

  elif (round_mode == "ibm_stochastic"):
    # This implementation is based on the IBM paper:
    # https://arxiv.org/pdf/1812.08011.pdf

    convert_x = tf.bitcast(x, tf.int32)

    # Need to take the floor
    mask = int(TRUNCATE_MASK)
    remain_mask = 0x1
    for j in range(0, num_mantissa):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    for j in range(1,reduce_num):
      remain_mask = remain_mask << 1
      remain_mask = remain_mask | 0x1

    #print("remain_mask: ", remain_mask)
    truncate_value = bitwise_ops.bitwise_and(convert_x, mask)

    # remain_mask masks the lower bits being truncated
    # no sign or exponent bits are retained
    remainder_value = bitwise_ops.bitwise_and(convert_x, remain_mask)
    #print("remain_value: ", remainder_value)

    round_exponent = bitwise_ops.bitwise_and(convert_x,EXP_MASK)
    round_exponent = bitwise_ops.right_shift(round_exponent, 23)
    exp_save = round_exponent
    round_exponent = round_exponent - num_mantissa

    # check for denormal round value
    ones_tensor = tf.ones_like(round_exponent)
    denormal_round = tf.math.less(round_exponent, ones_tensor)
    zeros_tensor = tf.zeros_like(round_exponent)
    round_exponent = tf.where(denormal_round, zeros_tensor, round_exponent)

    round_exponent = bitwise_ops.left_shift(round_exponent,23)
    round_sign = bitwise_ops.bitwise_and(convert_x,SIGN_MASK)
    
    round_value = bitwise_ops.bitwise_or(round_exponent,round_sign)

    float_round = tf.bitcast(round_value, tf.float32)
    float_x = tf.bitcast(truncate_value, tf.float32)
    #print("Truncated float_x: ", float_x)

    added_value = float_x + float_round
    #print("Added value : ", added_value)

    # need to move MSB of remainder  to the implicit one position of bit 23
    # adjust the exponent based on how many bits moved over
    # use cast instead of bitcast

    ############################

    float_remainder = tf.cast(remainder_value, tf.float32)
    # float_remainder is (m - truncated(m))

    # epsilon corresponds to e=2^k = value of round up by 1
    epsilon = (2**(reduce_num + 1))

    probability = float_remainder / epsilon
    #print("prob: ", probability)
    #print("probability: ", probability)

    random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)
    #print("random_prob: ", random_prob)
    random_prob = random_prob + probability
    #print("random_prob + prob: ", random_prob)

    ones_tensor = tf.ones_like(random_prob)
    final_probability = tf.math.greater_equal(random_prob, ones_tensor)
    #print("final_probability: ", final_probability)
    
    #final_probability = tf.cond((probability + random_prob) > 1.0,
    #                            lambda: True, lambda: False)

    final_value = tf.where(final_probability, added_value, float_x)

    exp_bits = bitwise_ops.bitwise_and(convert_x,EXP_MASK)
    # detect denormal values by looking for 1 in exponent
    zeroes_tensor = tf.zeros_like(exp_bits)
    denormal = tf.math.equal(exp_bits, zeroes_tensor)

    final_value = tf.where(denormal, float_x, final_value)

    x = final_value

    t_value = x

  elif (round_mode == "fast_stochastic"):
    # This implementation is based on the HPCA paper:
    # https://arxiv.org/pdf/2110.15456.pdf
    # This stochastic rounding method adds both + and - values
    # for the block of values, each value has an unique stochastic noise added
    

    convert_x = tf.bitcast(x, tf.int32)

    # Need to take the floor
    mask = int(TRUNCATE_MASK)
    remain_mask = 0x1
    for j in range(0, num_mantissa):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    for j in range(1,reduce_num):
      remain_mask = remain_mask << 1
      remain_mask = remain_mask | 0x1

    #print("remain_mask: ", remain_mask)
    truncate_value = bitwise_ops.bitwise_and(convert_x, mask)
    remainder_value = bitwise_ops.bitwise_and(convert_x, remain_mask)
    #print("remain_value: ", remainder_value)

    round_exponent = bitwise_ops.bitwise_and(convert_x,EXP_MASK)
    round_exponent = bitwise_ops.right_shift(round_exponent, 23)
    exp_save = round_exponent
    round_exponent = round_exponent - num_mantissa

    # check for denormal round value
    ones_tensor = tf.ones_like(round_exponent)
    denormal_round = tf.math.less(round_exponent, ones_tensor)
    zeros_tensor = tf.zeros_like(round_exponent)
    round_exponent = tf.where(denormal_round, zeros_tensor, round_exponent)

    round_exponent = bitwise_ops.left_shift(round_exponent,23)
    round_sign = bitwise_ops.bitwise_and(convert_x,SIGN_MASK)
    
    # round_value needs to be random for each value of the block
    # round_value needs to be either + or -
    # range of magnitude for round_value should be 0 to round bit
    round_value = bitwise_ops.bitwise_or(round_exponent,round_sign)

    pos_float_round = tf.bitcast(round_value, tf.float32)
    neg_float_round = round_value
    neg_float_round = bitwise_ops.bitwise_or(round_value,SIGN_MASK)
    neg_float_round = tf.bitcast(neg_float_round, tf.float32)

    float_round = tf.random.uniform(shape = x.shape, minval = neg_float_round, maxval = pos_float_round, dtype=tf.float32)

    float_x = tf.bitcast(truncate_value, tf.float32)
    #print("Truncated float_x: ", float_x)
    
    added_value = float_x + float_round

    convert_added_value = tf.bitcast(added_value, tf.int32)
    truncate_added_value = bitwise_ops.bitwise_and(convert_added_value, mask)
    added_value = tf.bitcast(truncate_added_value, tf.float32)

    x = added_value

    t_value = x

  elif (round_mode == "msfp_stochastic"):

    # Use the same exponent per tensor
    # Truncate bits that are shifted off - TODO: Look into rounding

    convert_x = tf.bitcast(x, tf.int32)

    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)

    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)
    
    # print("x: ", x, flush=True)
    # print("max_exp: ", max_exp, flush=True)

    saved_exp_bits = exp_bits
    saved_man_bits = man_bits
    # max_tensor = tf.ones_like(exp_bits)
    # max_tensor = tf.math.multiply(max_exp, max_tensor)
    diff_tensor = tf.math.subtract(max_exp,exp_bits)

    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 

    # Based on the difference of the exponent, round the bits shifted - start with truncation
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)

    # Need to mask out the bits lower than the num_mantissa bits
    mask = int(TRUNCATE_MASK)
    for j in range(0, num_mantissa-1):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    mask = tf.cast(mask, tf.int32)

    man_bits = bitwise_ops.bitwise_and(man_bits, mask)

    man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)

    final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    final_value = bitwise_ops.bitwise_or(final_value, man_bits)

    zeros_tensor = tf.zeros_like(man_bits)
    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)

    final_value = bitwise_ops.bitwise_and(final_value, mask)

    x_temp = tf.bitcast(final_value, tf.float32)

    ##################
    # Compute round up value
    # if entire value is truncated, need to round up to smallest representable bit

    guard_exp_bits = tf.subtract(saved_exp_bits, num_mantissa -1)
    guard_exp_bits = tf.add(guard_exp_bits, diff_tensor)

    # If exp of round up is smallest than the lowest representable bit, then fix to min
    smallest_exp = tf.subtract(max_exp, num_mantissa -1)
    complete_truncation = tf.less(guard_exp_bits, smallest_exp)
    # implicit_left = tf.equal(guard_exp_bits, smallest_exp)
    guard_exp_bits = tf.where(complete_truncation, smallest_exp, guard_exp_bits)

    guard_exp_bits = bitwise_ops.left_shift(guard_exp_bits, 23)
    round_up = bitwise_ops.bitwise_or(guard_exp_bits, sign_bit)

    ########## New Code for round up value

    guard_exp_bits = tf.subtract(max_exp, num_mantissa-1)
    # print("round_up_exp: ", guard_exp_bits)
    guard_exp_bits = bitwise_ops.left_shift(guard_exp_bits, 23)
    round_up = bitwise_ops.bitwise_or(guard_exp_bits, sign_bit)

    
    y = tf.bitcast(round_up, tf.float32)
    y = tf.where(tf.less(guard_exp_bits, 0), 0.0, y)

    #####################
    # Stochastic MSFP
    #####################

    # remain_mask should be the inverted version of the regular bit mask
    # reduce num, remain_mask, and random_prob need to be tensors
    # TODO: for block floating point, need to take care of case where
    # entire value is truncated. Implicit 1 needs to be factored in for the 
    # probability calculation also.

    # need to add implicit 1
    remain_mask = bitwise_ops.invert(mask)
    new_man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)
    new_man_bits = bitwise_ops.bitwise_or(new_man_bits, 0x00800000)
    # print("new_man_bits before shift: ", new_man_bits)
    new_man_bits = bitwise_ops.right_shift(new_man_bits,diff_tensor)
    remainder_value = bitwise_ops.bitwise_and(new_man_bits, remain_mask)
    # print("new_man_bits: ", new_man_bits)
    # print("diff_tensor: ", diff_tensor)
    # print("remain_mask: ", remain_mask)
    float_remainder = tf.cast(remainder_value, tf.float32)

    # greater_eq_range = tf.greater_equal(diff_tensor,num_mantissa - 1)

    reduce_num = 23
    # reduce_num = tf.repeat(reduce_num, tf.size(x))
    # reduce_num = tf.reshape(reduce_num, x.shape)
    # final diff is the position for round value - this is fixed for msfp
    # for afp, it varies depending on num_mantissa. always start with 24 - number of bits
    final_diff = num_mantissa - 1
    reduce_num = tf.subtract(reduce_num, final_diff)


    base = 2
    # base = tf.repeat(base, tf.size(x))
    # base = tf.reshape(base, x.shape)
    epsilon_int = tf.math.pow(base, reduce_num)
    epsilon_int = tf.cast(epsilon_int, tf.int32)
    epsilon_int = epsilon_int + 1

    '''
    epsilon = tf.cast(epsilon_int, tf.float32)
    epsilon = tf.repeat(epsilon, tf.size(x))
    epsilon = tf.reshape(epsilon, x.shape)
    # epsilon = (2**(reduce_num + 1))


    # probability = float_remainder / epsilon
    probability = tf.divide(float_remainder, epsilon)
    #print("prob: ", probability)
    #print("probability: ", probability)
    probability = tf.where(tf.equal(x, 0.0), 0.0, probability)
    # probability = tf.where(tf.logical_or(complete_truncation, implicit_left), 0.0, probability)

    random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)

    random_prob = random_prob + probability
    #print("random_prob + prob: ", random_prob)
    
    ones_tensor = tf.ones_like(random_prob)
    final_probability = tf.math.greater_equal(random_prob, ones_tensor)
    '''
    
    random_int = tf.random.uniform(shape = x.shape, maxval = epsilon_int, dtype=tf.int32)
    # random_int = tf.cast(random_int, tf.float32)
    remainder_value = tf.where(tf.equal(x, 0.0), 0, remainder_value)

    random_prob_int = tf.add(random_int, remainder_value)

    final_probability = tf.math.greater_equal(random_prob_int, epsilon_int)


    # print("max_exp: ", max_exp)
    # print("epsilon_int: ", epsilon_int)
    # print("remainder_value: ", remainder_value)
    # print("random_int: ", random_int)
    # print("random_prob_int: ", random_prob_int)

    x_temp = tf.where(final_probability, x_temp + y, x_temp)
    x_temp = tf.where(tf.equal(max_exp,0), x, x_temp)
    # print("x+y (rounded x): ", x_temp)
    # print("x: ", x)
    # print("y: ", y)    
    ######## Take care of denormal values

    # zeros_tensor = tf.zeros_like(exp_bits)
    # denormal = tf.math.equal(exp_bits, zeros_tensor)



    t_value = x_temp


  elif (round_mode == "msfp_nearest"):

    # Use the same exponent per tensor
    # Truncate bits that are shifted off - TODO: Look into rounding

    convert_x = tf.bitcast(x, tf.int32)

    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)

    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)
    
    # print("x: ", x, flush=True)
    # print("max_exp: ", max_exp, flush=True)

    saved_exp_bits = exp_bits
    saved_man_bits = man_bits
    # max_tensor = tf.ones_like(exp_bits)
    # max_tensor = tf.math.multiply(max_exp, max_tensor)
    diff_tensor = tf.math.subtract(max_exp,exp_bits)

    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 

    # Based on the difference of the exponent, round the bits shifted - start with truncation
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)

    # Need to mask out the bits lower than the num_mantissa bits
    mask = int(TRUNCATE_MASK)
    for j in range(0, num_mantissa-1):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    mask = tf.cast(mask, tf.int32)

    man_bits = bitwise_ops.bitwise_and(man_bits, mask)

    man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)

    final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    final_value = bitwise_ops.bitwise_or(final_value, man_bits)

    zeros_tensor = tf.zeros_like(man_bits)
    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)

    final_value = bitwise_ops.bitwise_and(final_value, mask)

    x_temp = tf.bitcast(final_value, tf.float32)

    ##################
    # Compute round up value
    # if entire value is truncated, need to round up to smallest representable bit

    guard_exp_bits = tf.subtract(saved_exp_bits, num_mantissa -1)
    guard_exp_bits = tf.add(guard_exp_bits, diff_tensor)

    # If exp of round up is smallest than the lowest representable bit, then fix to min
    smallest_exp = tf.subtract(max_exp, num_mantissa -1)
    complete_truncation = tf.less(guard_exp_bits, smallest_exp)
    # implicit_left = tf.equal(guard_exp_bits, smallest_exp)
    guard_exp_bits = tf.where(complete_truncation, smallest_exp, guard_exp_bits)

    guard_exp_bits = bitwise_ops.left_shift(guard_exp_bits, 23)
    round_up = bitwise_ops.bitwise_or(guard_exp_bits, sign_bit)

    ########## New Code for round up value

    guard_exp_bits = tf.subtract(max_exp, num_mantissa-1)
    # print("round_up_exp: ", guard_exp_bits)
    guard_exp_bits = bitwise_ops.left_shift(guard_exp_bits, 23)
    round_up = bitwise_ops.bitwise_or(guard_exp_bits, sign_bit)

    
    y = tf.bitcast(round_up, tf.float32)
    y = tf.where(tf.less(guard_exp_bits, 0), 0.0, y)

    #####################
    # Stochastic MSFP
    #####################

    # remain_mask should be the inverted version of the regular bit mask
    # reduce num, remain_mask, and random_prob need to be tensors
    # TODO: for block floating point, need to take care of case where
    # entire value is truncated. Implicit 1 needs to be factored in for the 
    # probability calculation also.

    # need to add implicit 1
    remain_mask = bitwise_ops.invert(mask)
    new_man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)
    new_man_bits = bitwise_ops.bitwise_or(new_man_bits, 0x00800000)
    new_man_bits = bitwise_ops.right_shift(new_man_bits,diff_tensor)
    remainder_value = bitwise_ops.bitwise_and(new_man_bits, remain_mask)

    # float_remainder = tf.cast(remainder_value, tf.float32)

    # greater_eq_range = tf.greater_equal(diff_tensor,num_mantissa - 1)

    reduce_num = 23
    # reduce_num = tf.repeat(reduce_num, tf.size(x))
    # reduce_num = tf.reshape(reduce_num, x.shape)
    # final diff is the position for round value - this is fixed for msfp
    # for afp, it varies depending on num_mantissa. always start with 24 - number of bits
    final_diff = num_mantissa 
    # removed -1 from num_mantissa to find round bit
    reduce_num = tf.subtract(reduce_num, final_diff)
    

    #print("float_remainder: ", float_remainder)
    # reduce_num = tf.add(reduce_num, 1)
    base = 2
    # base = tf.repeat(base, tf.size(x))
    # base = tf.reshape(base, x.shape)
    epsilon_int = tf.math.pow(base, reduce_num)
    epsilon_int = tf.cast(epsilon_int, tf.int32)
    # epsilon_int = epsilon_int + 1

    round_up_one = tf.greater_equal(remainder_value, epsilon_int)

    # print("x: ", x)
    # print("x_temp: ", x_temp)
    # print("y: ", y)
    # print("max_exp: ", max_exp)
    # print("epsilon_int: ", epsilon_int)
    # print("remainder_value: ", remainder_value)
    # print("round_up_one: ", round_up_one)
    # print("random_int: ", random_int)
    # print("random_prob_int: ", random_prob_int)
    x_temp = tf.where(round_up_one, x_temp + y, x_temp)
    # print("x+y: ", x_temp)
    ######## Take care of denormal values

    # zeros_tensor = tf.zeros_like(exp_bits)
    # denormal = tf.math.equal(exp_bits, zeros_tensor)



    t_value = x_temp


  elif (round_mode == "msfp_stochastic_v2"):

    # Use the same exponent per tensor
    # Truncate bits that are shifted off - TODO: Look into rounding

    convert_x = tf.bitcast(x, tf.int32)

    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)

    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)

    saved_exp_bits = exp_bits
    saved_man_bits = man_bits
    # max_tensor = tf.ones_like(exp_bits)
    # max_tensor = tf.math.multiply(max_exp, max_tensor)
    diff_tensor = tf.math.subtract(max_exp,exp_bits)

    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 

    # Based on the difference of the exponent, round the bits shifted - start with truncation
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)

    # Need to mask out the bits lower than the num_mantissa bits
    mask = int(TRUNCATE_MASK)
    for j in range(0, num_mantissa-1):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    mask = tf.cast(mask, tf.int32)

    man_bits = bitwise_ops.bitwise_and(man_bits, mask)

    man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)

    final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    final_value = bitwise_ops.bitwise_or(final_value, man_bits)

    zeros_tensor = tf.zeros_like(man_bits)
    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)

    final_value = bitwise_ops.bitwise_and(final_value, mask)

    x = tf.bitcast(final_value, tf.float32)

    ##################
    # Compute round up value

    guard_exp_bits = tf.subtract(saved_exp_bits, num_mantissa -1)
    guard_exp_bits = tf.add(guard_exp_bits, diff_tensor)
    guard_exp_bits = bitwise_ops.left_shift(guard_exp_bits, 23)
    round_up = bitwise_ops.bitwise_or(guard_exp_bits, sign_bit)
    # y = tf.bitcast(round_up, tf.float32)


    ##################

    pos_float_round = tf.bitcast(round_up, tf.float32)
    neg_float_round = round_up
    neg_float_round = bitwise_ops.bitwise_or(neg_float_round,SIGN_MASK)
    neg_float_round = tf.bitcast(neg_float_round, tf.float32)

    float_round = tf.random.uniform(shape = x.shape, minval = neg_float_round, maxval = pos_float_round, dtype=tf.float32)

    # float_x = tf.bitcast(truncate_value, tf.float32)
    #print("Truncated float_x: ", float_x)
    
    added_value = x + float_round

    convert_added_value = tf.bitcast(added_value, tf.int32)
    truncate_added_value = bitwise_ops.bitwise_and(convert_added_value, mask)
    added_value = tf.bitcast(truncate_added_value, tf.float32)


    x = added_value


    ##################

    t_value = x

  elif (round_mode == "msfp_prune_vs_quantize"):

    # Use the same exponent per tensor
    # Truncate bits that are shifted off - TODO: Look into rounding

    convert_x = tf.bitcast(x, tf.int32)

    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)

    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)

    # max_tensor = tf.ones_like(exp_bits)
    # max_tensor = tf.math.multiply(max_exp, max_tensor)
    diff_tensor = tf.math.subtract(max_exp,exp_bits)

    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 

    # Based on the difference of the exponent, round the bits shifted - start with truncation
    # exp_bits = bitwise_ops.left_shift(exp_bits,23)
    # man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)

    # # Need to mask out the bits lower than the num_mantissa bits
    # mask = int(TRUNCATE_MASK)
    # for j in range(0, num_mantissa-1):
    #   mask = mask >> 1
    #   mask = mask | 0x80000000
    
    # man_bits = bitwise_ops.bitwise_and(man_bits, mask)

    # man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)

    # final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    # final_value = bitwise_ops.bitwise_or(final_value, man_bits)

    zeros_tensor = tf.zeros_like(man_bits)
    final_value = tf.where(shift_to_zero, zeros_tensor, convert_x)

    # final_value = bitwise_ops.bitwise_and(final_value, mask)

    x = tf.bitcast(final_value, tf.float32)

    t_value = x


  elif (round_mode == "msfp_exponent_reuse"):

    # Use the same exponent per tensor
    # Truncate bits that are shifted off - TODO: Look into rounding

    convert_x = tf.bitcast(x, tf.int32)

    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)

    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)

    # max_tensor = tf.ones_like(exp_bits)
    # max_tensor = tf.math.multiply(max_exp, max_tensor)
    diff_tensor = tf.math.subtract(max_exp,exp_bits)

    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 

    # Based on the difference of the exponent, round the bits shifted - start with truncation
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)

    # Need to mask out the bits lower than the num_mantissa bits
    mask = int(TRUNCATE_MASK)
    for j in range(0, num_mantissa-1):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    man_bits = bitwise_ops.bitwise_and(man_bits, mask)

    man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)

    final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    final_value = bitwise_ops.bitwise_or(final_value, man_bits)

    zeros_tensor = tf.zeros_like(man_bits)
    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)

    final_value = bitwise_ops.bitwise_and(final_value, mask)

    x = tf.bitcast(final_value, tf.float32)

    t_value = x


  elif (round_mode == "msfp_pruning_2blocks"):

    # Use the same exponent per tensor
    # Truncate bits that are shifted off - TODO: Look into rounding

    convert_x = tf.bitcast(x, tf.int32)

    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)

    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)

    # max_tensor = tf.ones_like(exp_bits)
    # max_tensor = tf.math.multiply(max_exp, max_tensor)
    diff_tensor = tf.math.subtract(max_exp,exp_bits)

    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 

    # Based on the difference of the exponent, round the bits shifted - start with truncation
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)

    # Need to mask out the bits lower than the num_mantissa bits
    mask = int(TRUNCATE_MASK)
    for j in range(0, num_mantissa-1):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    man_bits = bitwise_ops.bitwise_and(man_bits, mask)
    
    # saving this for pruning code
    # saved_man_bits = man_bits

    man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)

    final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    final_value = bitwise_ops.bitwise_or(final_value, man_bits)

    zeros_tensor = tf.zeros_like(man_bits)
    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)

    final_value = bitwise_ops.bitwise_and(final_value, mask)

    x = tf.bitcast(final_value, tf.float32)

    t_value = x

    ##########################
    # pruning percentage code
    num_zeros = tf.math.count_nonzero(shift_to_zero)
    total_num = tf.size(shift_to_zero)
    num_zeros = tf.cast(num_zeros, tf.int32)
    average_zeros = (num_zeros*100/total_num)
    print("% of pruned zeros in all values: ", average_zeros.numpy())

    value_equal_zero = tf.math.equal(x, 0.0)
    total_zeros = tf.where(shift_to_zero, True, value_equal_zero)
    num_zeros = tf.math.count_nonzero(total_zeros)
    num_zeros = tf.cast(num_zeros, tf.int32)
    average_zeros = (num_zeros*100/total_num)
    print("% of total zeros in all values: ", average_zeros.numpy())

    # flatten the blocks to rows - similar to coarsening approach
    # reshape shift_to_zeros into groups of 4 values, groups of 2 values, and see
    # tf.print(shift_to_zero, summarize=64)
    # tf.print(shift_to_zero, summarize=64,output_stream=sys.stdout)


    # reshape shift_to_zeros into groups of 4 values, groups of 2 values, and see
    flatten_shift_to_zero = tf.reshape(shift_to_zero,[-1])

    num_blocks = total_num // 32
    remainder = total_num % 32
    if remainder:
      num_blocks += 1
      zero_padding = tf.zeros([remainder], tf.float32)
      zero_padding = tf.math.equal(zero_padding,0.0)
      flatten_shift_to_zero = numpy.concatenate((flatten_shift_to_zero, zero_padding), axis=None )

    groups_of_32 = tf.reshape(flatten_shift_to_zero, [num_blocks, 32])

    groups_of_32 = tf.math.count_nonzero(groups_of_32,1, keepdims=True )
    groups_of_32 = tf.math.greater_equal(groups_of_32,16)

    groups_of_32 = tf.math.count_nonzero(groups_of_32)
    num_blocks = tf.cast(num_blocks, tf.int64)
    percentage = groups_of_32*100/num_blocks

    print("Compression of 2 consecutive 16-value blocks %", percentage.numpy())

   # reshape shift_to_zeros into groups of 4 values, groups of 2 values, and see
    flatten_shift_to_zero = tf.reshape(shift_to_zero,[-1])
    compress_size = 16
    num_blocks = total_num // compress_size
    remainder = total_num % compress_size
    if remainder:
      num_blocks += 1
      zero_padding = tf.zeros([remainder], tf.float32)
      zero_padding = tf.math.equal(zero_padding,0.0)
      flatten_shift_to_zero = numpy.concatenate((flatten_shift_to_zero, zero_padding), axis=None )

    groups_of_32 = tf.reshape(flatten_shift_to_zero, [num_blocks, compress_size])

    groups_of_32 = tf.math.count_nonzero(groups_of_32,1, keepdims=True )
    groups_of_32 = tf.math.greater_equal(groups_of_32,compress_size//2)

    groups_of_32 = tf.math.count_nonzero(groups_of_32)
    num_blocks = tf.cast(num_blocks, tf.int64)
    percentage = groups_of_32*100/num_blocks

    print("Compression of 2 consecutive 8-value blocks %", percentage.numpy())

   # reshape shift_to_zeros into groups of 4 values, groups of 2 values, and see
    flatten_shift_to_zero = tf.reshape(shift_to_zero,[-1])
    compress_size = 8
    num_blocks = total_num // compress_size
    remainder = total_num % compress_size
    if remainder:
      num_blocks += 1
      zero_padding = tf.zeros([remainder], tf.float32)
      zero_padding = tf.math.equal(zero_padding,0.0)
      flatten_shift_to_zero = numpy.concatenate((flatten_shift_to_zero, zero_padding), axis=None )

    groups_of_32 = tf.reshape(flatten_shift_to_zero, [num_blocks, compress_size])

    groups_of_32 = tf.math.count_nonzero(groups_of_32,1, keepdims=True )
    groups_of_32 = tf.math.greater_equal(groups_of_32,compress_size//2)

    groups_of_32 = tf.math.count_nonzero(groups_of_32)
    num_blocks = tf.cast(num_blocks, tf.int64)
    percentage = groups_of_32*100/num_blocks

    print("Compression of 2 consecutive 4-value blocks %", percentage.numpy())

    flatten_shift_to_zero = tf.reshape(shift_to_zero,[-1])

    num_blocks = total_num // 4
    
    groups_of_2 = tf.reshape(flatten_shift_to_zero, [num_blocks, 4])

    groups_of_2 = tf.math.count_nonzero(groups_of_2,1, keepdims=True )
    groups_of_2 = tf.math.greater_equal(groups_of_2,2)

    groups_of_2 = tf.math.count_nonzero(groups_of_2)
    num_blocks = tf.cast(num_blocks, tf.int64)
    percentage = groups_of_2*100/num_blocks

    print("Compression of 2 consecutive 2-value blocks %", percentage.numpy())

    flatten_shift_to_zero = tf.reshape(shift_to_zero,[-1])

    num_blocks = total_num // 2
    
    groups_of_2 = tf.reshape(flatten_shift_to_zero, [num_blocks, 2])

    groups_of_2 = tf.math.count_nonzero(groups_of_2,1, keepdims=True )
    # if 1, value is pruned
    # <=1 pruned, 
    groups_of_2 = tf.math.greater_equal(groups_of_2,1)

    groups_of_2 = tf.math.count_nonzero(groups_of_2)
    num_blocks = tf.cast(num_blocks, tf.int64)
    percentage = groups_of_2*100/num_blocks

    print("Compression of 2 consecutive 1-value blocks %", percentage.numpy())


    # # Including trivialization
    # num_zeros = tf.math.count_nonzero(shift_to_zero)
    # total_num = tf.size(shift_to_zero)
    # num_zeros = tf.cast(num_zeros, tf.int32)
    # average_zeros = (num_zeros*100/total_num)

    # temp_x = tf.bitcast(x, tf.int32)
    # man_bits = bitwise_ops.bitwise_and(temp_x, MAN_MASK)    
    # trivial_two = tf.math.equal(man_bits, 0)
    # trivial_two = tf.reshape(trivial_two,[-1])
    # # print("type info: ", type(two_equal_two), type(trivial_two))
    # # trivial_two = tf.where(flatten_shift_to_zero,True, trivial_two)
    # flatten_shift_to_zero = tf.where(flatten_shift_to_zero, True, trivial_two)

    # num_blocks = total_num // 2
    # groups_of_two = tf.reshape(flatten_shift_to_zero, [num_blocks, 2])
    # num_blocks = total_num // 4
    # groups_of_four = tf.reshape(flatten_shift_to_zero, [num_blocks, 4])
    # groups_of_two = tf.math.count_nonzero(groups_of_two,1, keepdims=True )
    # groups_of_four = tf.math.count_nonzero(groups_of_four,1, keepdims=True )
    # groups_of_two = tf.reshape(groups_of_two, [-1])
    # groups_of_four = tf.reshape(groups_of_four, [-1])


    # two_equal_zero = tf.math.equal(groups_of_two, 0)
    # two_equal_one = tf.math.equal(groups_of_two, 1)
    # two_equal_two = tf.math.equal(groups_of_two, 2)

  
    # # print(tf.size(two_equal_two),tf.size(temp_x), flush=True)
  
    # # trivial_two = tf.math.count_nonzero(trivial_two)

    # two_equal_zero = tf.math.count_nonzero(two_equal_zero)
    # two_equal_one = tf.math.count_nonzero(two_equal_one)
    # two_equal_two = tf.math.count_nonzero(two_equal_two)

    # four_equal_zero = tf.math.equal(groups_of_four, 0)
    # four_equal_one = tf.math.equal(groups_of_four, 1)
    # four_equal_two = tf.math.equal(groups_of_four, 2)
    # four_equal_three = tf.math.equal(groups_of_four, 3)
    # four_equal_four = tf.math.equal(groups_of_four, 4)

    # four_equal_zero = tf.math.count_nonzero(four_equal_zero)
    # four_equal_one = tf.math.count_nonzero(four_equal_one)
    # four_equal_two = tf.math.count_nonzero(four_equal_two)
    # four_equal_three = tf.math.count_nonzero(four_equal_three)
    # four_equal_four = tf.math.count_nonzero(four_equal_four)

    # total_num = tf.cast(total_num, tf.int64)
    # # print("groups of 2 - percentage of 1, 2 + trivial: ", (two_equal_one + two_equal_two)*100/(total_num//2), "- pruned 0: ", two_equal_zero, "1: ", two_equal_one, "2: ", two_equal_two)
    # # print("groups of 4 - percentage of 3, 4 + trivial: ", (four_equal_three + four_equal_four)*100/(total_num//4),"- pruned 0: ", four_equal_zero, "1: ", four_equal_one, "2: ", four_equal_two, "3: ", four_equal_three, "4: ", four_equal_four)
    # # print("groups of 2 - percentage of 1, 2 + trivial: ", (two_equal_one + two_equal_two)*100/(total_num//2))
    # print("groups of 4 - percentage of 3, 4 + trivial: ", (four_equal_three + four_equal_four)*100/(total_num//4))


  elif (round_mode == "msfp_lottery_vs_quantize_20"):

    # Use the same exponent per tensor
    # Truncate bits that are shifted off - TODO: Look into rounding

    save_x = x

    convert_x = tf.bitcast(x, tf.int32)

    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)

    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)

    # max_tensor = tf.ones_like(exp_bits)
    # max_tensor = tf.math.multiply(max_exp, max_tensor)
    diff_tensor = tf.math.subtract(max_exp,exp_bits)

    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 

    # Based on the difference of the exponent, round the bits shifted - start with truncation
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)

    # Need to mask out the bits lower than the num_mantissa bits
    mask = int(TRUNCATE_MASK)
    for j in range(0, num_mantissa-1):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    man_bits = bitwise_ops.bitwise_and(man_bits, mask)
    
    # saving this for pruning code
    # saved_man_bits = man_bits

    man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)

    final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    final_value = bitwise_ops.bitwise_or(final_value, man_bits)

    zeros_tensor = tf.zeros_like(man_bits)
    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)

    final_value = bitwise_ops.bitwise_and(final_value, mask)

    x = tf.bitcast(final_value, tf.float32)

    t_value = x

    ##########################
    # pruning percentage code
    quantize_mask = shift_to_zero
    num_zeros = tf.math.count_nonzero(shift_to_zero)
    total_num = tf.size(shift_to_zero)
    num_to_prune = tf.cast(total_num, tf.float32) * 0.2
    num_to_prune = tf.cast(num_to_prune, tf.int64)
    lottery_mask = tf.zeros_like(shift_to_zero)
    lottery_mask = tf.where(True, False, False)
    pruned_num = 0
    max_value = tf.math.reduce_max(save_x)

    # p20 = tfp.stats.percentile(save_x, q=20)
    # print("p20", p20)
    # lotter_mask = tf.where(tf.math.less_equal(save_x, p20), True, False)
    # print("lottery_mask",lottery_mask)
    # print("num_to_prune", num_to_prune)
    print("total, num_to_prune", total_num.numpy(), num_to_prune.numpy())
    while (pruned_num < num_to_prune):
      min_value = tf.math.reduce_min(save_x)
      lottery_mask = tf.where(tf.math.less_equal(save_x,min_value), True, lottery_mask)
      save_x = tf.where(lottery_mask, max_value, save_x)
      pruned_num = tf.math.count_nonzero(lottery_mask)
      # print("pruned_num", pruned_num)

    pruned_num = tf.math.count_nonzero(lottery_mask)
    # lottery_mask of 20% pruning is created
    inverted_pruned = tf.math.logical_not(lottery_mask)
    diff_mask = tf.where(inverted_pruned, quantize_mask, False)
    diff_num = tf.math.count_nonzero(diff_mask)
    print("Per-layer pruning stats - num_pruned_quantize:", num_zeros.numpy(), ":num_pruned_lottery:",  pruned_num.numpy(), ":num additional pruning:", diff_num.numpy(),":\% diff:", diff_num.numpy()/pruned_num.numpy(), flush=True)


  elif (round_mode == "msfp_lottery_vs_quantize_50"):

    # Use the same exponent per tensor
    # Truncate bits that are shifted off - TODO: Look into rounding

    save_x = x

    convert_x = tf.bitcast(x, tf.int32)

    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)

    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)

    # max_tensor = tf.ones_like(exp_bits)
    # max_tensor = tf.math.multiply(max_exp, max_tensor)
    diff_tensor = tf.math.subtract(max_exp,exp_bits)

    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 

    # Based on the difference of the exponent, round the bits shifted - start with truncation
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)

    # Need to mask out the bits lower than the num_mantissa bits
    mask = int(TRUNCATE_MASK)
    for j in range(0, num_mantissa-1):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    man_bits = bitwise_ops.bitwise_and(man_bits, mask)
    
    # saving this for pruning code
    # saved_man_bits = man_bits

    man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)

    final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    final_value = bitwise_ops.bitwise_or(final_value, man_bits)

    zeros_tensor = tf.zeros_like(man_bits)
    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)

    final_value = bitwise_ops.bitwise_and(final_value, mask)

    x = tf.bitcast(final_value, tf.float32)

    t_value = x

    ##########################
    # pruning percentage code
    quantize_mask = shift_to_zero
    num_zeros = tf.math.count_nonzero(shift_to_zero)
    total_num = tf.size(shift_to_zero)
    num_to_prune = tf.cast(total_num, tf.float32) * 0.5
    num_to_prune = tf.cast(num_to_prune, tf.int64)
    lottery_mask = tf.zeros_like(shift_to_zero)
    pruned_num = 0
    max_value = tf.math.reduce_max(save_x)

    # p20 = tfp.stats.percentile(save_x, q=20)
    # print("p20", p20)
    # lotter_mask = tf.where(tf.math.less_equal(save_x, p20), True, False)
    # print("lottery_mask",lottery_mask)
    # print("num_to_prune", num_to_prune)
    print("total, num_to_prune", total_num.numpy(), num_to_prune.numpy())
    while (pruned_num < num_to_prune):
      min_value = tf.math.reduce_min(save_x)
      lottery_mask = tf.where(tf.math.less_equal(save_x,min_value), True, lottery_mask)
      save_x = tf.where(lottery_mask, max_value, save_x)
      pruned_num = tf.math.count_nonzero(lottery_mask)
      # print("pruned_num", pruned_num)

    pruned_num = tf.math.count_nonzero(lottery_mask)
    # lottery_mask of 20% pruning is created
    inverted_pruned = tf.math.logical_not(lottery_mask)
    diff_mask = tf.where(inverted_pruned, quantize_mask, False)
    diff_num = tf.math.count_nonzero(diff_mask)
    print("Per-layer pruning stats - num_pruned_quantize:", num_zeros.numpy(), ":num_pruned_lottery:",  pruned_num.numpy(), ":num additional pruning:", diff_num.numpy(),":\% diff:", diff_num.numpy()/pruned_num.numpy(), flush=True)


  elif (round_mode == "msfp_pruning"):

    # Use the same exponent per tensor
    # Truncate bits that are shifted off - TODO: Look into rounding

    convert_x = tf.bitcast(x, tf.int32)

    sign_bit = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)

    exp_bits = bitwise_ops.right_shift(exp_bits,23)
    max_exp = tf.math.reduce_max(exp_bits, 1, keepdims=True)

    # max_tensor = tf.ones_like(exp_bits)
    # max_tensor = tf.math.multiply(max_exp, max_tensor)
    diff_tensor = tf.math.subtract(max_exp,exp_bits)

    shift_to_zero = tf.math.greater(diff_tensor, num_mantissa-1) 

    # Based on the difference of the exponent, round the bits shifted - start with truncation
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    man_bits = bitwise_ops.right_shift(man_bits, diff_tensor)

    # Need to mask out the bits lower than the num_mantissa bits
    mask = int(TRUNCATE_MASK)
    for j in range(0, num_mantissa-1):
      mask = mask >> 1
      mask = mask | 0x80000000
    
    man_bits = bitwise_ops.bitwise_and(man_bits, mask)
    
    # saving this for pruning code
    # saved_man_bits = man_bits

    man_bits = bitwise_ops.left_shift(man_bits, diff_tensor)

    final_value = bitwise_ops.bitwise_or(sign_bit,exp_bits)
    final_value = bitwise_ops.bitwise_or(final_value, man_bits)

    zeros_tensor = tf.zeros_like(man_bits)
    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)

    final_value = bitwise_ops.bitwise_and(final_value, mask)

    x = tf.bitcast(final_value, tf.float32)

    t_value = x

    ##########################
    # pruning percentage code
    num_zeros = tf.math.count_nonzero(shift_to_zero)
    total_num = tf.size(shift_to_zero)
    num_zeros = tf.cast(num_zeros, tf.int32)
    average_zeros = (num_zeros*100/total_num)
    print("% zeros: ", average_zeros)

    # reshape shift_to_zeros into groups of 4 values, groups of 2 values, and see
    flatten_shift_to_zero = tf.reshape(shift_to_zero,[-1])

    num_blocks = total_num // 2
    groups_of_two = tf.reshape(flatten_shift_to_zero, [num_blocks, 2])
    num_blocks = total_num // 4
    groups_of_four = tf.reshape(flatten_shift_to_zero, [num_blocks, 4])
    groups_of_two = tf.math.count_nonzero(groups_of_two,1, keepdims=True )
    groups_of_four = tf.math.count_nonzero(groups_of_four,1, keepdims=True )
    groups_of_two = tf.reshape(groups_of_two, [-1])
    groups_of_four = tf.reshape(groups_of_four, [-1])


    two_equal_zero = tf.math.equal(groups_of_two, 0)
    two_equal_one = tf.math.equal(groups_of_two, 1)
    two_equal_two = tf.math.equal(groups_of_two, 2)

  
    # print(tf.size(two_equal_two),tf.size(temp_x), flush=True)
  
    # trivial_two = tf.math.count_nonzero(trivial_two)

    two_equal_zero = tf.math.count_nonzero(two_equal_zero)
    two_equal_one = tf.math.count_nonzero(two_equal_one)
    two_equal_two = tf.math.count_nonzero(two_equal_two)

    four_equal_zero = tf.math.equal(groups_of_four, 0)
    four_equal_one = tf.math.equal(groups_of_four, 1)
    four_equal_two = tf.math.equal(groups_of_four, 2)
    four_equal_three = tf.math.equal(groups_of_four, 3)
    four_equal_four = tf.math.equal(groups_of_four, 4)

    four_equal_zero = tf.math.count_nonzero(four_equal_zero)
    four_equal_one = tf.math.count_nonzero(four_equal_one)
    four_equal_two = tf.math.count_nonzero(four_equal_two)
    four_equal_three = tf.math.count_nonzero(four_equal_three)
    four_equal_four = tf.math.count_nonzero(four_equal_four)

    total_num = tf.cast(total_num, tf.int64)
    # print("groups of 2 - percentage of 1, 2: ", (two_equal_one + two_equal_two)*100/(total_num//2), "- pruned 0: ", two_equal_zero, "1: ", two_equal_one, "2: ", two_equal_two)
    # print("groups of 4 - percentage of 3, 4: ", (four_equal_three + four_equal_four)*100/(total_num//4),"- pruned 0: ", four_equal_zero, "1: ", four_equal_one, "2: ", four_equal_two, "3: ", four_equal_three, "4: ", four_equal_four)
    # print("groups of 2 - percentage of 1, 2: ", (two_equal_one + two_equal_two)*100/(total_num//2))
    print("groups of 4 - percentage of 3, 4: ", (four_equal_three + four_equal_four)*100/(total_num//4))

    # Including trivialization
    num_zeros = tf.math.count_nonzero(shift_to_zero)
    total_num = tf.size(shift_to_zero)
    num_zeros = tf.cast(num_zeros, tf.int32)
    average_zeros = (num_zeros*100/total_num)

    temp_x = tf.bitcast(x, tf.int32)
    man_bits = bitwise_ops.bitwise_and(temp_x, MAN_MASK)    
    trivial_two = tf.math.equal(man_bits, 0)
    trivial_two = tf.reshape(trivial_two,[-1])
    # print("type info: ", type(two_equal_two), type(trivial_two))
    # trivial_two = tf.where(flatten_shift_to_zero,True, trivial_two)
    flatten_shift_to_zero = tf.where(flatten_shift_to_zero, True, trivial_two)

    num_blocks = total_num // 2
    groups_of_two = tf.reshape(flatten_shift_to_zero, [num_blocks, 2])
    num_blocks = total_num // 4
    groups_of_four = tf.reshape(flatten_shift_to_zero, [num_blocks, 4])
    groups_of_two = tf.math.count_nonzero(groups_of_two,1, keepdims=True )
    groups_of_four = tf.math.count_nonzero(groups_of_four,1, keepdims=True )
    groups_of_two = tf.reshape(groups_of_two, [-1])
    groups_of_four = tf.reshape(groups_of_four, [-1])


    two_equal_zero = tf.math.equal(groups_of_two, 0)
    two_equal_one = tf.math.equal(groups_of_two, 1)
    two_equal_two = tf.math.equal(groups_of_two, 2)

  
    # print(tf.size(two_equal_two),tf.size(temp_x), flush=True)
  
    # trivial_two = tf.math.count_nonzero(trivial_two)

    two_equal_zero = tf.math.count_nonzero(two_equal_zero)
    two_equal_one = tf.math.count_nonzero(two_equal_one)
    two_equal_two = tf.math.count_nonzero(two_equal_two)

    four_equal_zero = tf.math.equal(groups_of_four, 0)
    four_equal_one = tf.math.equal(groups_of_four, 1)
    four_equal_two = tf.math.equal(groups_of_four, 2)
    four_equal_three = tf.math.equal(groups_of_four, 3)
    four_equal_four = tf.math.equal(groups_of_four, 4)

    four_equal_zero = tf.math.count_nonzero(four_equal_zero)
    four_equal_one = tf.math.count_nonzero(four_equal_one)
    four_equal_two = tf.math.count_nonzero(four_equal_two)
    four_equal_three = tf.math.count_nonzero(four_equal_three)
    four_equal_four = tf.math.count_nonzero(four_equal_four)

    total_num = tf.cast(total_num, tf.int64)
    # print("groups of 2 - percentage of 1, 2 + trivial: ", (two_equal_one + two_equal_two)*100/(total_num//2), "- pruned 0: ", two_equal_zero, "1: ", two_equal_one, "2: ", two_equal_two)
    # print("groups of 4 - percentage of 3, 4 + trivial: ", (four_equal_three + four_equal_four)*100/(total_num//4),"- pruned 0: ", four_equal_zero, "1: ", four_equal_one, "2: ", four_equal_two, "3: ", four_equal_three, "4: ", four_equal_four)
    # print("groups of 2 - percentage of 1, 2 + trivial: ", (two_equal_one + two_equal_two)*100/(total_num//2))
    print("groups of 4 - percentage of 3, 4 + trivial: ", (four_equal_three + four_equal_four)*100/(total_num//4))


  elif (round_mode == "block_based_nearest_non_zero_sign_encoding_fix4_signfix_optimization3_v2"):
    # AFP with all positive bit, no unused bits
    
    # break up 32 float into individual components

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

    extra_bits = 0

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

    x = tf.bitcast(final_value, tf.float32)

    x = tf.where(half_bool, x + y, x)

    t_value = x



  elif (round_mode == "block_based_nearest_non_zero_sign_encoding_fix4_signfix_optimization3_v3"):
    # AFP with no positive bit, no unused bits
    
    # break up 32 float into individual components

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
    # num_mantissa = tf.where(all_one_sign, num_mantissa + 1, num_mantissa)

    num_mantissa_save = num_mantissa

    extra_bits = 0

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

    x = tf.bitcast(final_value, tf.float32)

    x = tf.where(half_bool, x + y, x)

    t_value = x



  elif (round_mode == "block_based_nearest_non_zero_sign_encoding_fix4_signfix_optimization3"):

    # break up 32 float into individual components

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
    # unused2_imp3 = tf.logical_and(unused2_imp3, tf.logical_not(all_one_sign))
    # unused2_imp4 = tf.logical_and(unused2_imp4, tf.logical_not(all_one_sign))    
    # unused2_imp5 = tf.logical_and(unused2_imp5, max_exp_sel)

    # num_mantissa = tf.where(unused2_imp1, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp2, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp3, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp4, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp5, num_mantissa + 1, num_mantissa) 

    # unused2_imp = tf.where(unused2_imp5, 1, 0 )
    # unused2_imp = tf.where(unused2_imp4, unused2_imp + 1, unused2_imp)
    # unused2_imp = tf.where(unused2_imp4, 1, 0 )
    # unused2_imp = tf.where(unused2_imp3, unused2_imp + 1, unused2_imp )
    # unused2_imp = tf.where(unused2_imp2, 1, 0 )
    # unused2_imp = tf.where(unused2_imp1, unused2_imp + 1, unused2_imp )

    # print("unused2_imp: ", unused2_imp)
    # Make use of the free encodings
    # num_mantissa_save2 = num_mantissa_save

    # unused2_imp1 = tf.logical_and(unused2_imp1, tf.greater_equal(diff_tensor, 2))
    # unused2_imp2 = tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 4))
    # unused2_imp3 = tf.logical_and(unused2_imp3, tf.equal(diff_tensor, 6))
    # unused2_imp4 = tf.logical_and(unused2_imp4, tf.equal(diff_tensor,8))
    # unused2_imp5 = tf.logical_and(unused2_imp5, tf.equal(diff_tensor,10))


    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 1), tf.equal(diff_tensor, 1)), 
    #                         num_mantissa + 1, num_mantissa) 
    
    #extra_bits = tf.where(tf.greater_equal(unused2_imp, 2), num_mantissa_save -5, 0)

    # shift_amount = diff_tensor

    # twos_tensor = tf.ones_like(man_bits)
    # twos_tensor = tf.add(twos_tensor, twos_tensor)
    # extra_bits = tf.where(tf.less(extra_bits, 0), 0, extra_bits)
    # powers_of_two = tf.math.pow(twos_tensor, extra_bits)
    # powers_of_two = tf.where(tf.less_equal(powers_of_two,1), 0, powers_of_two)
    # shift_to_zero = tf.where(tf.greater(diff_tensor, num_mantissa + powers_of_two))

    # shift_amount_to_zero = tf.where(tf.greater_equal(shift_amount, num_mantissa + powers_of_two),
    #                                 num_mantissa_save - 6, 0)
    
    # shift_amount_to_zero = tf.roll(shift_amount_to_zero, shift=-1, axis=1)


    # num_mantissa = tf.where(tf.greater_equal(unused2_imp, 5), 
    #                         num_mantissa + shift_amount_to_zero, num_mantissa)

    num_mantissa = tf.where(tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)), 
                            num_mantissa + 1, num_mantissa)
    
    num_mantissa = tf.where(tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)), 
                            num_mantissa + 1, num_mantissa)

    # num_mantissa = tf.where(tf.logical_and(unused2_imp3, tf.equal(diff_tensor, 2)), 
    #                         num_mantissa + 1, num_mantissa)

    # num_mantissa = tf.where(tf.logical_and(unused2_imp4, tf.equal(diff_tensor, 3)), 
    #                         num_mantissa + 1, num_mantissa)            
                                 
    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 2), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 

    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 3), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa)
                                 
    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, ), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 

    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 3), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 

    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 4), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 
    
    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 5), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 
    
    # num_mantissa = tf.where(unused2_imp4, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp5, num_mantissa + 1, num_mantissa)            
    # extra_bits gives you the number of bits left for exponent encoding

    # print("upper half num_mantissa: ", num_mantissa)
    # print("extra_bits: ", extra_bits)
    # extra_bits = tf.where(unused2_imp2, num_mantissa - 3, 0)

    # Use the extra bits for more accuracy
    # add_bit = tf.cast(extra_bits2, tf.bool)
    # add_bit = tf.where(tf.greater_equal(diff_tensor,unused2_imp), add_bit, False)

    #####################
    # Check to see if values can use the additional encodings
    # Code to check to see if values are eligible
    # add_bit1= tf.greater_equal(diff_tensor,1)
    # add_bit2= tf.greater_equal(diff_tensor,2)
    # add_bit3= tf.greater_equal(diff_tensor,3)
    # add_bit4= tf.greater_equal(diff_tensor,4)
    # add_bit5= tf.greater_equal(diff_tensor,5) 

    # select_add1 = tf.logical_and(add_bit1, unused2_imp1)    
    # select_add2 = tf.logical_and(add_bit2, unused2_imp2)  
    # select_add3 = tf.logical_and(add_bit3, unused2_imp3)  
    # select_add4 = tf.logical_and(add_bit4, unused2_imp4)  
    # select_add5 = tf.logical_and(add_bit5, unused2_imp5)

    # May want to reverse the ordering to prioritize interval 1
    # select_add = tf.where(select_add5, chain5_list, 0)
    # select_add = tf.where(select_add4, chain4_list, select_add)
    # select_add = tf.where(select_add3, chain3_list, select_add)
    # select_add = tf.where(select_add2, chain2_list, select_add)
    # select_add = tf.where(select_add1, chain1_list, select_add)
    # select_add = tf.where(select_add5, 5, 0)
    # select_add = tf.where(select_add4, 4, select_add)
    # select_add = tf.where(select_add3, 3, select_add)
    # select_add = tf.where(select_add2, 2, select_add)
    # select_add = tf.where(select_add1, 1, select_add)

    # Need to make sure there are enough bits to provide the encodings!
    # select_add_bool = tf.cast(select_add, tf.bool)
    # num_mantissa = tf.where(select_add_bool, num_mantissa + 1, num_mantissa)

    # num_mantissa = tf.where(add_extra_bit, num_mantissa + 1, num_mantissa)

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

    x = tf.bitcast(final_value, tf.float32)

    x = tf.where(half_bool, x + y, x)

    t_value = x

  elif (round_mode == "afp_optimization3_new_exponly_v2"):
    # This new mode takes the encoding of offset = 0b111 and mantissa 0bxxxx1 as exp only mode
    # so exp is stored in the first 4 bits of the mantissa field.
    
    # break up 32 float into individual components

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
    num_mantissa = tf.where(tf.greater_equal(diff_tensor,7), num_mantissa_save + 3, num_mantissa)

    extra_bits = num_mantissa_save - 3 - 1
    
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
    # unused2_imp3 = tf.logical_and(unused2_imp3, tf.logical_not(all_one_sign))
    # unused2_imp4 = tf.logical_and(unused2_imp4, tf.logical_not(all_one_sign))    
    # unused2_imp5 = tf.logical_and(unused2_imp5, max_exp_sel)

    # num_mantissa = tf.where(unused2_imp1, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp2, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp3, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp4, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp5, num_mantissa + 1, num_mantissa) 

    # unused2_imp = tf.where(unused2_imp5, 1, 0 )
    # unused2_imp = tf.where(unused2_imp4, unused2_imp + 1, unused2_imp)
    # unused2_imp = tf.where(unused2_imp4, 1, 0 )
    # unused2_imp = tf.where(unused2_imp3, unused2_imp + 1, unused2_imp )
    # unused2_imp = tf.where(unused2_imp2, 1, 0 )
    # unused2_imp = tf.where(unused2_imp1, unused2_imp + 1, unused2_imp )

    # print("unused2_imp: ", unused2_imp)
    # Make use of the free encodings
    # num_mantissa_save2 = num_mantissa_save

    # unused2_imp1 = tf.logical_and(unused2_imp1, tf.greater_equal(diff_tensor, 2))
    # unused2_imp2 = tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 4))
    # unused2_imp3 = tf.logical_and(unused2_imp3, tf.equal(diff_tensor, 6))
    # unused2_imp4 = tf.logical_and(unused2_imp4, tf.equal(diff_tensor,8))
    # unused2_imp5 = tf.logical_and(unused2_imp5, tf.equal(diff_tensor,10))


    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 1), tf.equal(diff_tensor, 1)), 
    #                         num_mantissa + 1, num_mantissa) 
    
    #extra_bits = tf.where(tf.greater_equal(unused2_imp, 2), num_mantissa_save -5, 0)

    # shift_amount = diff_tensor

    # twos_tensor = tf.ones_like(man_bits)
    # twos_tensor = tf.add(twos_tensor, twos_tensor)
    # extra_bits = tf.where(tf.less(extra_bits, 0), 0, extra_bits)
    # powers_of_two = tf.math.pow(twos_tensor, extra_bits)
    # powers_of_two = tf.where(tf.less_equal(powers_of_two,1), 0, powers_of_two)
    # shift_to_zero = tf.where(tf.greater(diff_tensor, num_mantissa + powers_of_two))

    # shift_amount_to_zero = tf.where(tf.greater_equal(shift_amount, num_mantissa + powers_of_two),
    #                                 num_mantissa_save - 6, 0)
    
    # shift_amount_to_zero = tf.roll(shift_amount_to_zero, shift=-1, axis=1)


    # num_mantissa = tf.where(tf.greater_equal(unused2_imp, 5), 
    #                         num_mantissa + shift_amount_to_zero, num_mantissa)

    num_mantissa = tf.where(tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)), 
                            num_mantissa + 1, num_mantissa)
    
    num_mantissa = tf.where(tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)), 
                            num_mantissa + 1, num_mantissa)

    # num_mantissa = tf.where(tf.logical_and(unused2_imp3, tf.equal(diff_tensor, 2)), 
    #                         num_mantissa + 1, num_mantissa)

    # num_mantissa = tf.where(tf.logical_and(unused2_imp4, tf.equal(diff_tensor, 3)), 
    #                         num_mantissa + 1, num_mantissa)            
                                 
    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 2), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 

    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 3), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa)
                                 
    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, ), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 

    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 3), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 

    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 4), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 
    
    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 5), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 
    
    # num_mantissa = tf.where(unused2_imp4, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp5, num_mantissa + 1, num_mantissa)            
    # extra_bits gives you the number of bits left for exponent encoding

    # print("upper half num_mantissa: ", num_mantissa)
    # print("extra_bits: ", extra_bits)
    # extra_bits = tf.where(unused2_imp2, num_mantissa - 3, 0)

    # Use the extra bits for more accuracy
    # add_bit = tf.cast(extra_bits2, tf.bool)
    # add_bit = tf.where(tf.greater_equal(diff_tensor,unused2_imp), add_bit, False)

    #####################
    # Check to see if values can use the additional encodings
    # Code to check to see if values are eligible
    # add_bit1= tf.greater_equal(diff_tensor,1)
    # add_bit2= tf.greater_equal(diff_tensor,2)
    # add_bit3= tf.greater_equal(diff_tensor,3)
    # add_bit4= tf.greater_equal(diff_tensor,4)
    # add_bit5= tf.greater_equal(diff_tensor,5) 

    # select_add1 = tf.logical_and(add_bit1, unused2_imp1)    
    # select_add2 = tf.logical_and(add_bit2, unused2_imp2)  
    # select_add3 = tf.logical_and(add_bit3, unused2_imp3)  
    # select_add4 = tf.logical_and(add_bit4, unused2_imp4)  
    # select_add5 = tf.logical_and(add_bit5, unused2_imp5)

    # May want to reverse the ordering to prioritize interval 1
    # select_add = tf.where(select_add5, chain5_list, 0)
    # select_add = tf.where(select_add4, chain4_list, select_add)
    # select_add = tf.where(select_add3, chain3_list, select_add)
    # select_add = tf.where(select_add2, chain2_list, select_add)
    # select_add = tf.where(select_add1, chain1_list, select_add)
    # select_add = tf.where(select_add5, 5, 0)
    # select_add = tf.where(select_add4, 4, select_add)
    # select_add = tf.where(select_add3, 3, select_add)
    # select_add = tf.where(select_add2, 2, select_add)
    # select_add = tf.where(select_add1, 1, select_add)

    # Need to make sure there are enough bits to provide the encodings!
    # select_add_bool = tf.cast(select_add, tf.bool)
    # num_mantissa = tf.where(select_add_bool, num_mantissa + 1, num_mantissa)

    # num_mantissa = tf.where(add_extra_bit, num_mantissa + 1, num_mantissa)

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
    # round_exp_only = bitwise_ops.bitwise_and(saved_man_bits, 0x00700000)
    # round_exp_only = tf.cast(round_exp_only, tf.bool)
    exp_only_value = bitwise_ops.bitwise_and(convert_x, 0xFF800000)
    # exp_only_value = tf.where(round_exp_only, tf.add(saved_exp_bits, 1), saved_exp_bits)
    # exp_only_value = bitwise_ops.left_shift(exp_only_value, 23)
    #exp_only_value = bitwise_ops.bitwise_or(exp_only_value, sign_bit)

    final_value = tf.where(shift_to_zero, zeros_tensor, final_value)
    final_value = tf.where(shift_to_zero_round_up, round_up_value, final_value)
    final_value = tf.where(shift_to_zero_round_up_exp, exp_only_value, final_value)

    # Final check to make sure we are only using bits we have
    final_value = bitwise_ops.bitwise_and(final_value, mask)

    x = tf.bitcast(final_value, tf.float32)

    x = tf.where(half_bool, x + y, x)

    t_value = x

  elif (round_mode == "afp_optimization3_new_exponly"):
    # This new mode takes the encoding of offset = 0b111 and mantissa 0bxxxx1 as exp only mode
    # so exp is stored in the first 4 bits of the mantissa field.
    
    # break up 32 float into individual components

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
    num_mantissa = tf.where(tf.greater_equal(diff_tensor,7), num_mantissa_save + 3, num_mantissa)

    extra_bits = num_mantissa_save - 3 - 1
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
    # unused2_imp3 = tf.logical_and(unused2_imp3, tf.logical_not(all_one_sign))
    # unused2_imp4 = tf.logical_and(unused2_imp4, tf.logical_not(all_one_sign))    
    # unused2_imp5 = tf.logical_and(unused2_imp5, max_exp_sel)

    # num_mantissa = tf.where(unused2_imp1, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp2, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp3, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp4, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp5, num_mantissa + 1, num_mantissa) 

    # unused2_imp = tf.where(unused2_imp5, 1, 0 )
    # unused2_imp = tf.where(unused2_imp4, unused2_imp + 1, unused2_imp)
    # unused2_imp = tf.where(unused2_imp4, 1, 0 )
    # unused2_imp = tf.where(unused2_imp3, unused2_imp + 1, unused2_imp )
    # unused2_imp = tf.where(unused2_imp2, 1, 0 )
    # unused2_imp = tf.where(unused2_imp1, unused2_imp + 1, unused2_imp )

    # print("unused2_imp: ", unused2_imp)
    # Make use of the free encodings
    # num_mantissa_save2 = num_mantissa_save

    # unused2_imp1 = tf.logical_and(unused2_imp1, tf.greater_equal(diff_tensor, 2))
    # unused2_imp2 = tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 4))
    # unused2_imp3 = tf.logical_and(unused2_imp3, tf.equal(diff_tensor, 6))
    # unused2_imp4 = tf.logical_and(unused2_imp4, tf.equal(diff_tensor,8))
    # unused2_imp5 = tf.logical_and(unused2_imp5, tf.equal(diff_tensor,10))


    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 1), tf.equal(diff_tensor, 1)), 
    #                         num_mantissa + 1, num_mantissa) 
    
    #extra_bits = tf.where(tf.greater_equal(unused2_imp, 2), num_mantissa_save -5, 0)

    # shift_amount = diff_tensor

    # twos_tensor = tf.ones_like(man_bits)
    # twos_tensor = tf.add(twos_tensor, twos_tensor)
    # extra_bits = tf.where(tf.less(extra_bits, 0), 0, extra_bits)
    # powers_of_two = tf.math.pow(twos_tensor, extra_bits)
    # powers_of_two = tf.where(tf.less_equal(powers_of_two,1), 0, powers_of_two)
    # shift_to_zero = tf.where(tf.greater(diff_tensor, num_mantissa + powers_of_two))

    # shift_amount_to_zero = tf.where(tf.greater_equal(shift_amount, num_mantissa + powers_of_two),
    #                                 num_mantissa_save - 6, 0)
    
    # shift_amount_to_zero = tf.roll(shift_amount_to_zero, shift=-1, axis=1)


    # num_mantissa = tf.where(tf.greater_equal(unused2_imp, 5), 
    #                         num_mantissa + shift_amount_to_zero, num_mantissa)

    num_mantissa = tf.where(tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)), 
                            num_mantissa + 1, num_mantissa)
    
    num_mantissa = tf.where(tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)), 
                            num_mantissa + 1, num_mantissa)

    # num_mantissa = tf.where(tf.logical_and(unused2_imp3, tf.equal(diff_tensor, 2)), 
    #                         num_mantissa + 1, num_mantissa)

    # num_mantissa = tf.where(tf.logical_and(unused2_imp4, tf.equal(diff_tensor, 3)), 
    #                         num_mantissa + 1, num_mantissa)            
                                 
    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 2), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 

    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 3), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa)
                                 
    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, ), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 

    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 3), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 

    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 4), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 
    
    # num_mantissa = tf.where(tf.logical_and(tf.greater_equal(unused2_imp, 5), 
    #                                        tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa) 
    
    # num_mantissa = tf.where(unused2_imp4, num_mantissa + 1, num_mantissa)
    # num_mantissa = tf.where(unused2_imp5, num_mantissa + 1, num_mantissa)            
    # extra_bits gives you the number of bits left for exponent encoding

    # print("upper half num_mantissa: ", num_mantissa)
    # print("extra_bits: ", extra_bits)
    # extra_bits = tf.where(unused2_imp2, num_mantissa - 3, 0)

    # Use the extra bits for more accuracy
    # add_bit = tf.cast(extra_bits2, tf.bool)
    # add_bit = tf.where(tf.greater_equal(diff_tensor,unused2_imp), add_bit, False)

    #####################
    # Check to see if values can use the additional encodings
    # Code to check to see if values are eligible
    # add_bit1= tf.greater_equal(diff_tensor,1)
    # add_bit2= tf.greater_equal(diff_tensor,2)
    # add_bit3= tf.greater_equal(diff_tensor,3)
    # add_bit4= tf.greater_equal(diff_tensor,4)
    # add_bit5= tf.greater_equal(diff_tensor,5) 

    # select_add1 = tf.logical_and(add_bit1, unused2_imp1)    
    # select_add2 = tf.logical_and(add_bit2, unused2_imp2)  
    # select_add3 = tf.logical_and(add_bit3, unused2_imp3)  
    # select_add4 = tf.logical_and(add_bit4, unused2_imp4)  
    # select_add5 = tf.logical_and(add_bit5, unused2_imp5)

    # May want to reverse the ordering to prioritize interval 1
    # select_add = tf.where(select_add5, chain5_list, 0)
    # select_add = tf.where(select_add4, chain4_list, select_add)
    # select_add = tf.where(select_add3, chain3_list, select_add)
    # select_add = tf.where(select_add2, chain2_list, select_add)
    # select_add = tf.where(select_add1, chain1_list, select_add)
    # select_add = tf.where(select_add5, 5, 0)
    # select_add = tf.where(select_add4, 4, select_add)
    # select_add = tf.where(select_add3, 3, select_add)
    # select_add = tf.where(select_add2, 2, select_add)
    # select_add = tf.where(select_add1, 1, select_add)

    # Need to make sure there are enough bits to provide the encodings!
    # select_add_bool = tf.cast(select_add, tf.bool)
    # num_mantissa = tf.where(select_add_bool, num_mantissa + 1, num_mantissa)

    # num_mantissa = tf.where(add_extra_bit, num_mantissa + 1, num_mantissa)

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

    x = tf.bitcast(final_value, tf.float32)

    x = tf.where(half_bool, x + y, x)

    t_value = x


  elif (round_mode == "afp_stochastic"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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

    x = tf.bitcast(final_value, tf.float32)

    # x = tf.where(half_bool, x + y, x)

    #####################
    # Stochastic AFP
    #####################

    # remain_mask should be the inverted version of the regular bit mask
    # reduce num, remain_mask, and random_prob need to be tensors

    remain_mask = bitwise_ops.invert(mask)
    reduce_num = 23
    reduce_num = tf.repeat(reduce_num, tf.size(x))
    reduce_num = tf.reshape(reduce_num, x.shape)
    reduce_num = tf.subtract(reduce_num, num_mantissa)
    # reduce_num = 23 - num_mantissa

    # for j in range(1,reduce_num):
    #   remain_mask = remain_mask << 1
    #   remain_mask = remain_mask | 0x1

    remainder_value = bitwise_ops.bitwise_and(convert_x, remain_mask)

    float_remainder = tf.cast(remainder_value, tf.float32)
    #print("float_remainder: ", float_remainder)
    reduce_num = tf.add(reduce_num, 1)
    base = 2
    base = tf.repeat(base, tf.size(x))
    base = tf.reshape(base, x.shape)
    epsilon = tf.math.pow(base, reduce_num)
    epsilon = tf.cast(epsilon, tf.float32)
    # epsilon = (2**(reduce_num + 1))

    # probability = float_remainder / epsilon
    probability = tf.divide(float_remainder, epsilon)
    #print("prob: ", probability)
    #print("probability: ", probability)

    random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)

    #print("random_prob: ", random_prob)
    random_prob = random_prob + probability
    #print("random_prob + prob: ", random_prob)

    ones_tensor = tf.ones_like(random_prob)
    final_probability = tf.math.greater(random_prob, ones_tensor)

    x = tf.where(final_probability, x + y, x)
    

    t_value = x


  elif (round_mode == "afp_stochastic_v3_share"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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

    x = tf.bitcast(final_value, tf.float32)

    # x = tf.where(half_bool, x + y, x)

    # Stochastic AFP
    # remain_mask should be the inverted version of the regular bit mask
    # reduce num, remain_mask, and random_prob need to be tensors

    temp_diff = tf.where(tf.greater(diff_tensor, num_mantissa), num_mantissa, diff_tensor)
    temp_mask = bitwise_ops.left_shift(mask,temp_diff)
    remain_mask = bitwise_ops.invert(temp_mask)
    reduce_num = 23
    reduce_num = tf.repeat(reduce_num, tf.size(x))
    reduce_num = tf.reshape(reduce_num, x.shape)
    reduce_num = tf.subtract(reduce_num, num_mantissa)
    # reduce_num = 23 - num_mantissa

    # for j in range(1,reduce_num):
    #   remain_mask = remain_mask << 1
    #   remain_mask = remain_mask | 0x1

    remainder_value = bitwise_ops.bitwise_and(convert_x, remain_mask)

    remainder_value = bitwise_ops.right_shift(remainder_value, diff_tensor)
    float_remainder = tf.cast(remainder_value, tf.float32)
    #print("float_remainder: ", float_remainder)
    reduce_num = tf.add(reduce_num, 1)
    base = 2
    base = tf.repeat(base, tf.size(x))
    base = tf.reshape(base, x.shape)
    epsilon = tf.math.pow(base, reduce_num)
    epsilon = tf.cast(epsilon, tf.float32)
    # epsilon = (2**(reduce_num + 1))

    # probability = float_remainder / epsilon
    probability = tf.divide(float_remainder, epsilon)
    #print("prob: ", probability)
    #print("probability: ", probability)

    # random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)
    random_prob = tf.random.uniform(maxval = 1, dtype=tf.float32)
    random_prob = tf.repeat(random_prob, tf.size(x))
    random_prob = tf.reshape(random_prob, x.shape)

    #print("random_prob: ", random_prob)
    random_prob = random_prob + probability

    random_prob = tf.where(tf.equal(x,0.0), 0, random_prob)
    #print("random_prob + prob: ", random_prob)

    ones_tensor = tf.ones_like(random_prob)
    final_probability = tf.math.greater(random_prob, ones_tensor)

    x = tf.where(final_probability, x + y, x)
    
  
    t_value = x


  elif (round_mode == "afp_stochastic_v3_new2"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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

    temp_diff = tf.where(tf.greater(diff_tensor, num_mantissa), num_mantissa, diff_tensor)
    temp_mask = bitwise_ops.left_shift(mask,temp_diff)
    remain_mask = bitwise_ops.invert(temp_mask)
    reduce_num = 23
    reduce_num = tf.repeat(reduce_num, tf.size(x))
    reduce_num = tf.reshape(reduce_num, x.shape)
    reduce_num = tf.subtract(reduce_num, num_mantissa)
    # reduce_num = 23 - num_mantissa

    # for j in range(1,reduce_num):
    #   remain_mask = remain_mask << 1
    #   remain_mask = remain_mask | 0x1

    remainder_value = bitwise_ops.bitwise_and(convert_x, remain_mask)

    remainder_value = bitwise_ops.right_shift(remainder_value, diff_tensor)
    float_remainder = tf.cast(remainder_value, tf.float32)
    #print("float_remainder: ", float_remainder)
    reduce_num = tf.add(reduce_num, 1)
    base = 2
    base = tf.repeat(base, tf.size(x))
    base = tf.reshape(base, x.shape)
    epsilon = tf.math.pow(base, reduce_num)
    epsilon = tf.cast(epsilon, tf.float32)
    # epsilon = (2**(reduce_num + 1))

    # probability = float_remainder / epsilon
    probability = tf.divide(float_remainder, epsilon)
    #print("prob: ", probability)
    #print("probability: ", probability)

    random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)

    #print("random_prob: ", random_prob)
    random_prob = tf.add(random_prob, probability)

    random_prob = tf.where(tf.equal(x,0.0), 0, random_prob)
    #print("random_prob + prob: ", random_prob)

    ones_tensor = tf.ones_like(random_prob)
    final_probability = tf.math.greater_equal(random_prob, ones_tensor)

    # final_value = tf.where(final_probability, x + y, x)
    
    x_temp = tf.where(final_probability, x_temp + y, x_temp)

    x_temp = tf.where(tf.equal(max_exp,0), x, x_temp)
    ##########################
    # Denormal values
    ##########################

    # exp_bits = bitwise_ops.bitwise_and(convert_x,EXP_MASK)
    # # detect denormal values by looking for 1 in exponent
    # zeroes_tensor = tf.zeros_like(exp_bits)
    # denormal = tf.math.equal(exp_bits, zeroes_tensor)

    # x = tf.where(denormal, x, final_value) 
        

    t_value = x_temp

  elif (round_mode == "afp_stochastic_v3"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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

    temp_diff = tf.where(tf.greater(diff_tensor, num_mantissa), num_mantissa, diff_tensor)
    temp_mask = bitwise_ops.left_shift(mask,temp_diff)
    remain_mask = bitwise_ops.invert(temp_mask)
    reduce_num = 23
    reduce_num = tf.repeat(reduce_num, tf.size(x))
    reduce_num = tf.reshape(reduce_num, x.shape)
    reduce_num = tf.subtract(reduce_num, num_mantissa)
    # reduce_num = 23 - num_mantissa

    # for j in range(1,reduce_num):
    #   remain_mask = remain_mask << 1
    #   remain_mask = remain_mask | 0x1

    remainder_value = bitwise_ops.bitwise_and(convert_x, remain_mask)

    remainder_value = bitwise_ops.right_shift(remainder_value, diff_tensor)
    float_remainder = tf.cast(remainder_value, tf.float32)
    #print("float_remainder: ", float_remainder)
    reduce_num = tf.add(reduce_num, 1)
    base = 2
    base = tf.repeat(base, tf.size(x))
    base = tf.reshape(base, x.shape)
    epsilon = tf.math.pow(base, reduce_num)
    epsilon = tf.cast(epsilon, tf.float32)
    # epsilon = (2**(reduce_num + 1))

    # probability = float_remainder / epsilon
    probability = tf.divide(float_remainder, epsilon)
    #print("prob: ", probability)
    #print("probability: ", probability)

    random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)

    #print("random_prob: ", random_prob)
    random_prob = random_prob + probability

    random_prob = tf.where(tf.equal(x,0.0), 0, random_prob)
    #print("random_prob + prob: ", random_prob)

    ones_tensor = tf.ones_like(random_prob)
    final_probability = tf.math.greater_equal(random_prob, ones_tensor)

    # final_value = tf.where(final_probability, x + y, x)
    
    x_temp = tf.where(final_probability, x_temp + y, x_temp)

    x_temp = tf.where(tf.equal(max_exp,0), x, x_temp)
    ##########################
    # Denormal values
    ##########################

    # exp_bits = bitwise_ops.bitwise_and(convert_x,EXP_MASK)
    # # detect denormal values by looking for 1 in exponent
    # zeroes_tensor = tf.zeros_like(exp_bits)
    # denormal = tf.math.equal(exp_bits, zeroes_tensor)

    # x = tf.where(denormal, x, final_value) 
        

    t_value = x_temp


  elif (round_mode == "afp_stochastic_v3_new"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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

    temp_diff = tf.where(tf.greater(diff_tensor, num_mantissa), num_mantissa, diff_tensor)
    temp_mask = bitwise_ops.left_shift(mask,temp_diff)
    remain_mask = bitwise_ops.invert(temp_mask)
    reduce_num = 23
    reduce_num = tf.repeat(reduce_num, tf.size(x))
    reduce_num = tf.reshape(reduce_num, x.shape)
    reduce_num = tf.subtract(reduce_num, num_mantissa)
    # reduce_num = 23 - num_mantissa

    # for j in range(1,reduce_num):
    #   remain_mask = remain_mask << 1
    #   remain_mask = remain_mask | 0x1

    remainder_value = bitwise_ops.bitwise_and(convert_x, remain_mask)

    remainder_value = bitwise_ops.right_shift(remainder_value, diff_tensor)
    float_remainder = tf.cast(remainder_value, tf.float32)
    #print("float_remainder: ", float_remainder)
    reduce_num = tf.add(reduce_num, 1)
    base = 2
    base = tf.repeat(base, tf.size(x))
    base = tf.reshape(base, x.shape)
    epsilon = tf.math.pow(base, reduce_num)
    epsilon = tf.cast(epsilon, tf.float32)
    # epsilon = (2**(reduce_num + 1))

    # probability = float_remainder / epsilon
    probability = tf.divide(float_remainder, epsilon)

    d_morris = epsilon
    probability = tf.divide(1, d_morris)
    #print("prob: ", probability)
    #print("probability: ", probability)

    random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)

    #print("random_prob: ", random_prob)
    # random_prob = random_prob + probability

    # random_prob = tf.where(tf.equal(x,0.0), 0, random_prob)
    #print("random_prob + prob: ", random_prob)

    ones_tensor = tf.ones_like(random_prob)
    final_probability = tf.math.greater(d_morris, random_prob)

    # final_value = tf.where(final_probability, x + y, x)
    
    # print("remain_mask: ", remain_mask)
    # print("remainder_value: ", remainder_value)
    # print("float_remainder", float_remainder)
    # print("epsilon: ", epsilon)
    # print("probability:", probability)
    # print("random prob: ", random_prob)
    # print("final prob: ", final_probability)


    x_temp = tf.where(final_probability, x_temp + y, x_temp)
    x_temp = tf.where(tf.equal(max_exp,0), x, x_temp)
    # print("x+y or x (rounded): ", x_temp )
    # print("x: ", x)
    # print("y: ", y)
    ##########################
    # Denormal values
    ##########################

    # exp_bits = bitwise_ops.bitwise_and(convert_x,EXP_MASK)
    # # detect denormal values by looking for 1 in exponent
    # zeroes_tensor = tf.zeros_like(exp_bits)
    # denormal = tf.math.equal(exp_bits, zeroes_tensor)

    # x = tf.where(denormal, x, final_value) 
        

    t_value = x_temp


  elif (round_mode == "afp_stochastic_v4"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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
    new_shift = tf.where(greater_than_7, tf.where(all_one_sign, num_mantissa_save, num_mantissa_save - 1), new_shift )
 
    # Using arithmetic right shift to add 1's on the left side of the mask
    new_mask = tf.bitwise.right_shift(new_mask, new_shift)
    # new_mask_random = tf.bitwise.right_shift(tf.cast(TRUNCATE_MASK, tf.int32), new_shift - 1)

    remain_mask = bitwise_ops.invert(new_mask)
    # print("remain_mask: ", remain_mask)
    # random_mask = bitwise_ops.invert(new_mask_random)
    # print("random_mask: ", random_mask)

    new_man_bits = bitwise_ops.bitwise_and(convert_x, MAN_MASK)
    new_man_bits = bitwise_ops.bitwise_or(new_man_bits, 0x00800000)
    new_man_bits_greater_7 = bitwise_ops.right_shift(new_man_bits,diff_tensor-7)
    new_man_bits = tf.where(greater_than_7, new_man_bits_greater_7, new_man_bits)
    remainder_value = bitwise_ops.bitwise_and(new_man_bits, remain_mask)

    reduce_num = 23
    final_diff = actual_mantissa - 1
    # diff < 7
    reduce_num = tf.subtract(reduce_num, final_diff)
    # diff >= 7
    reduce_num = tf.where(tf.logical_or(equal_7, greater_than_7), 23 - (num_mantissa - 7 - 1),reduce_num)


    base = 2
    epsilon_int = tf.math.pow(base, reduce_num)
    epsilon_int = tf.cast(epsilon_int, tf.int32)

    # New epsilon_int code
    epsilon_int = remain_mask + 1

    epsilon_int = epsilon_int + 1
    # epsilon_int_max = tf.reduce_max(epsilon_int)
    epsilon_int_max = tf.math.pow(2,23)
    epsilon_int_max += 1
    
    random_int = tf.random.uniform(shape = x.shape, maxval = epsilon_int_max, dtype=tf.int32)
    random_int = bitwise_ops.right_shift(random_int, new_shift-1)

    # random_int = bitwise_ops.bitwise_and(random_int, random_mask)
    random_int = bitwise_ops.right_shift(random_int, tf.reduce_max(reduce_num) - reduce_num)
    # random_int = tf.cast(random_int, tf.float32)
    remainder_value = tf.where(tf.equal(x, 0.0), 0, remainder_value)

    random_prob_int = tf.add(random_int, remainder_value)

    final_probability = tf.math.greater_equal(random_prob_int, epsilon_int)

    print("max_exp: ", max_exp -127)
    print("diff: ", diff_tensor)
    print("all_one_sign: ", all_one_sign)
    print("unused 0: ", tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)))
    print("unused 1: ", tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)))
    print("num_mantissa: ", num_mantissa)
    print("reduce_num: ", reduce_num)
    print("epsilon_int: ", epsilon_int)
    print("remainder_value: ", remainder_value)
    print("random_int: ", random_int)
    print("random_prob_int: ", random_prob_int)

    x_temp = tf.where(final_probability, x_temp + y, x_temp)
    x_temp = tf.where(tf.equal(max_exp,0), x, x_temp)
    print("x+y or x (rounded): ", x_temp )
    print("x: ", x)
    print("y: ", y)
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


  elif (round_mode == "afp_nearest"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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


  elif (round_mode == "afp_nearest_dynamic_full"):
    # This is based on the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

    # dynamically determine when blocks need to be in bfloat 

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

    num_outlier = tf.math.count_nonzero((tf.greater(diff_tensor,11)))
    if num_outlier > 1:
      # round entire block to bfloat format - 1 sign, 8 exponent, 7 mantissa
      mask = int(NEAREST_PTR)
      # print("rounding to bfloat16! ", end = "")
      for j in range(1, 7):
        mask = mask >> 1
        mask = mask | NEAREST_PTR
      truncate_x = bitwise_ops.bitwise_and(convert_x, mask)
      
      mask2 = int(NEAREST_PTR)
      for j in range(0, 7):
        mask2 = mask2 >> 1
      half_one = bitwise_ops.bitwise_and(convert_x, mask2)
      half_bool = tf.cast(half_one, tf.bool)

      mantissa_bits = bitwise_ops.bitwise_and(truncate_x, MAN_MASK)
      rest_bits = bitwise_ops.bitwise_and(convert_x, SIGN_EXP_MASK)
      truncate_x = bitwise_ops.bitwise_or(rest_bits, mantissa_bits)
      pre_roundup_float = tf.bitcast(truncate_x, tf.float32)

      round_up_value = bitwise_ops.bitwise_and(convert_x, SIGN_MASK)
      exp_bits = exp_bits - 8
      exp_bits = bitwise_ops.left_shift(exp_bits,23)
      round_up_value = bitwise_ops.bitwise_or(round_up_value, exp_bits)
      round_up_float = tf.bitcast(round_up_value, tf.float32)

      final_float = tf.where(half_bool, round_up_float + pre_roundup_float, pre_roundup_float)
      t_value = final_float
      return t_value
  


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


  elif (round_mode == "afp_nearest_narrow"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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
    # num_mantissa = tf.where(all_one_sign, num_mantissa + 1, num_mantissa)

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


    # num_mantissa = tf.where(tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa)
    
    # num_mantissa = tf.where(tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)), 
    #                         num_mantissa + 1, num_mantissa)

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


  elif (round_mode == "afp_stochastic_v5"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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
    # new_man_bits = bitwise_ops.bitwise_or(new_man_bits, 0x00800000)
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

    # epsilon_int = epsilon_int + 1
    # epsilon_int_max = tf.reduce_max(epsilon_int)
    # epsilon_int_max = tf.math.pow(2,23)
    # epsilon_int_max += 1
    
    probability = tf.cast(tf.divide(remainder_value, epsilon_int), tf.float32)

    random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)

    # remainder_value = tf.where(tf.equal(x, 0.0), 0, remainder_value)

    # random_prob_int = tf.add(random_int, remainder_value)

    final_probability = tf.add(random_prob, probability)
    ones_tensor = tf.ones_like(random_prob)
    final_round = tf.math.greater_equal(final_probability, ones_tensor)


    # print("new_shift: ", new_shift)
    # print("max_exp: ", max_exp -127)
    # print("diff: ", diff_tensor)
    # print("all_one_sign: ", all_one_sign)
    # print("unused 0: ", tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)))
    # print("unused 1: ", tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)))
    # print("num_mantissa: ", num_mantissa)
    # print("reduce_num: ", reduce_num)
    # print("epsilon_int: ", epsilon_int)
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



  elif (round_mode == "afp_stochastic_narrow_v5"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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
    # num_mantissa = tf.where(all_one_sign, num_mantissa + 1, num_mantissa)

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


    # num_mantissa = tf.where(tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)), 
    #                         num_mantissa + 1, num_mantissa)
    
    # num_mantissa = tf.where(tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)), 
    #                         num_mantissa + 1, num_mantissa)

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
    # new_man_bits = bitwise_ops.bitwise_or(new_man_bits, 0x00800000)
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

    # epsilon_int = epsilon_int + 1
    # epsilon_int_max = tf.reduce_max(epsilon_int)
    # epsilon_int_max = tf.math.pow(2,23)
    # epsilon_int_max += 1
    
    probability = tf.cast(tf.divide(remainder_value, epsilon_int), tf.float32)

    random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)

    # remainder_value = tf.where(tf.equal(x, 0.0), 0, remainder_value)

    # random_prob_int = tf.add(random_int, remainder_value)

    final_probability = tf.add(random_prob, probability)
    ones_tensor = tf.ones_like(random_prob)
    final_round = tf.math.greater_equal(final_probability, ones_tensor)


    # print("new_shift: ", new_shift)
    # print("max_exp: ", max_exp -127)
    # print("diff: ", diff_tensor)
    # print("all_one_sign: ", all_one_sign)
    # print("unused 0: ", tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)))
    # print("unused 1: ", tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)))
    # print("num_mantissa: ", num_mantissa)
    # print("reduce_num: ", reduce_num)
    # print("epsilon_int: ", epsilon_int)
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


  elif (round_mode == "afp_stochastic_v6"):
    # This is the standard AFP rounding with positive and unused bits
    # Need to implement stochastic rounding for rounding
    # use 1 random value per block?
    # Can use the exponent as random seed?
    # break up 32 float into individual components

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
    new_man_bits = bitwise_ops.bitwise_or(new_man_bits, 0x00800000)
    new_man_bits_greater_7 = bitwise_ops.right_shift(new_man_bits,diff_tensor-7)
    new_man_bits = tf.where(greater_than_7, new_man_bits_greater_7, new_man_bits)
    remainder_value = bitwise_ops.bitwise_and(new_man_bits, remain_mask)

    reduce_num = 23
    final_diff = actual_mantissa - 1
    # diff < 7
    reduce_num = tf.subtract(reduce_num, final_diff)
    # diff >= 7
    reduce_num = tf.where(tf.logical_or(equal_7, greater_than_7), 23 - (num_mantissa - 7 - 1),reduce_num)


    base = 2
    epsilon_int = tf.math.pow(base, reduce_num)
    epsilon_int = tf.cast(epsilon_int, tf.int32)

    # New epsilon_int code
    epsilon_int = remain_mask + 1

    epsilon_int = epsilon_int + 1
    # epsilon_int_max = tf.reduce_max(epsilon_int)
    epsilon_int_max = tf.math.pow(2,23)
    epsilon_int_max += 1
    
    random_int = tf.random.uniform(shape = x.shape, maxval = epsilon_int_max, dtype=tf.int32)
    random_int = bitwise_ops.right_shift(random_int, new_shift)

    # random_int = bitwise_ops.bitwise_and(random_int, random_mask)
    random_int = bitwise_ops.right_shift(random_int, tf.reduce_max(reduce_num) - reduce_num)
    # random_int = tf.cast(random_int, tf.float32)
    remainder_value = tf.where(tf.equal(x, 0.0), 0, remainder_value)
  
    
    half_round = tf.greater(remainder_value, bitwise_ops.right_shift(epsilon_int-1,1)) 

    random_prob_int = tf.add(random_int, remainder_value)

    final_probability = tf.where(half_round, True, tf.math.greater_equal(random_prob_int, epsilon_int))

    # print("new_shift: ", new_shift)
    # print("max_exp: ", max_exp -127)
    # print("diff: ", diff_tensor)
    # print("all_one_sign: ", all_one_sign)
    # print("unused 0: ", tf.logical_and(unused2_imp1, tf.equal(diff_tensor, 0)))
    # print("unused 1: ", tf.logical_and(unused2_imp2, tf.equal(diff_tensor, 1)))
    # print("num_mantissa: ", num_mantissa)
    # print("reduce_num: ", reduce_num)
    # print("epsilon_int: ", epsilon_int)
    # print("remainder_value: ", remainder_value)
    # print("random_int: ", random_int)
    # print("random_prob_int: ", random_prob_int)

    x_temp = tf.where(final_probability, x_temp + y, x_temp)
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


  elif (round_mode == "afp_stochastic_v2"):
    # Need to take into account that the mantissa is shifted
    # by the diff_tensor amount
    # So the lower bits are being truncated with the right shift
    # in order for the stochastic rounding to be exact, 
    # Need to setup the remaining mask to hold all truncated values
    # and 1 needs to be the round up value

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

    x = tf.bitcast(final_value, tf.float32)

    # x = tf.where(half_bool, x + y, x)

    # Stochastic AFP
    # remain_mask should be the inverted version of the regular bit mask
    # reduce num, remain_mask, and random_prob need to be tensors

    # Need to add 1's to the shifted number of bits due to diff tensor
    remain_mask = bitwise_ops.invert(mask)

    add_remain_mask = tf.cast(0, tf.uint32)
    temp = (23-diff_tensor)
    temp = tf.cast(temp, tf.uint32)
    add_remain_mask = bitwise_ops.invert(add_remain_mask)
    add_remain_mask = tf.repeat(add_remain_mask, tf.size(x))
    add_remain_mask = tf.reshape(add_remain_mask, x.shape)
    add_remain_mask = tf.cast(add_remain_mask, tf.uint32)
    add_remain_mask = bitwise_ops.right_shift(add_remain_mask, 23 - diff_tensor)

    remain_mask = bitwise_ops.left_shift(remain_mask, diff_tensor)
    remain_mask = bitwise_ops.bitwise_or(remain_mask, add_remain_mask)


    reduce_num = 23
    reduce_num = tf.repeat(reduce_num, tf.size(x))
    reduce_num = tf.reshape(reduce_num, x.shape)
    reduce_num = tf.subtract(reduce_num, num_mantissa)
    reduce_num = tf.add(reduce_num, diff_tensor)
    # reduce_num = 23 - num_mantissa

    # for j in range(1,reduce_num):
    #   remain_mask = remain_mask << 1
    #   remain_mask = remain_mask | 0x1

    # 
    remainder_value = bitwise_ops.bitwise_and(convert_x, remain_mask)

    float_remainder = tf.cast(remainder_value, tf.float32)
    #print("float_remainder: ", float_remainder)
    reduce_num = tf.add(reduce_num, 1)
    base = 2
    base = tf.repeat(base, tf.size(x))
    base = tf.reshape(base, x.shape)
    epsilon = tf.math.pow(base, reduce_num)
    epsilon = tf.cast(epsilon, tf.float32)
    # epsilon = (2**(reduce_num + 1))

    # probability = float_remainder / epsilon
    probability = tf.divide(float_remainder, epsilon)
    #print("prob: ", probability)
    #print("probability: ", probability)

    random_prob = tf.random.uniform(shape = x.shape, maxval = 1, dtype=tf.float32)

    #print("random_prob: ", random_prob)
    random_prob = random_prob + probability
    #print("random_prob + prob: ", random_prob)

    ones_tensor = tf.ones_like(random_prob)
    final_probability = tf.math.greater(random_prob, ones_tensor)

    x = tf.where(final_probability, x + y, x)
    

    t_value = x

  elif (round_mode == "nearest"):
    # FIXME: Need to apply tf.where changes to the custom_round_not_tensor - or merge functionality

    # TODO: Need to fix the problem with there is a round up but it causes a carry into
    # bit position 23 - need to adjust both exponent and mantissa
    # This code rounds by doing the following:
    # 1) truncate to the correct # of bits
    # 2) check the bit to the right of the last valid bit
    # 3) If bit to the right (half_one) is 1, then need to round up
    # 4) round up by creating the correct value in floating point (with correct exp and use implicit 1 in mantissa)
    # 5) then doing an addition.
    # 6) In order to do this with the tensors, first mask off that half_one bit
    # 7) Then shift to the right to the 0th bit position
    # 8) cast this bit to float using tf.cast
    # 9) multiply the round value with this half_one bit/float
    # 10) add result to truncated value 

    mask = int(NEAREST_PTR)
    #print("This is the original mask: ",hex(mask))
    for j in range(1, num_mantissa):
      mask = mask >> 1
      mask = mask | NEAREST_PTR

    #print("This is the new mask: ", hex(mask))
    convert_x = tf.bitcast(x, tf.int32)
    #print("mantissa convert_x: ", bitwise_ops.bitwise_and(convert_x, MAN_MASK))
    truncate_x = bitwise_ops.bitwise_and(convert_x, mask)
    #print("truncate_x: ", truncate_x)

    mask2 = int(NEAREST_PTR)
    #mask_low = int(LOWEST_PTR)

    for j in range(0, num_mantissa):
      mask2 = mask2 >> 1
      # mask will have a 1 at the bit to the right of the last mantissa bit

    #print("This is mask2: ", hex(mask2))
    # half_one stores the bit to the right of the last mantissa bit
    half_one = bitwise_ops.bitwise_and(convert_x, mask2)
    half_one = bitwise_ops.right_shift(half_one, reduce_num -1)

    # Create a tensor of Boolean values
    half_bool = tf.cast(half_one, tf.bool)
    #print("half_bool:", half_bool)

    mantissa_bits = bitwise_ops.bitwise_and(truncate_x, MAN_MASK)

    rest_bits = bitwise_ops.bitwise_and(convert_x, SIGN_EXP_MASK)

    # Need to create a float for the round up value
    # then add to the floating point value
    # floating point implicit 1, so mantissa = 0
    
    exp_bits = bitwise_ops.bitwise_and(convert_x, EXP_MASK)
    exp_bits = bitwise_ops.right_shift(exp_bits,23)

    # detect denormal values by looking for 1 in exponent
    zeros_tensor = tf.zeros_like(exp_bits)
    denormal = tf.math.equal(exp_bits, zeros_tensor)

    #print("exp_bits: ", exp_bits)

    #################################################################
    # NOTE: This is - num_mantissa to move the binary point to the 
    # correct location for the round value
    ones_tensor = tf.ones_like(exp_bits)
    exp_bits = exp_bits - (num_mantissa * ones_tensor)
    denormal_round = tf.math.less(exp_bits,ones_tensor)

    #print("exp_bits: ", exp_bits)
    exp_bits = bitwise_ops.left_shift(exp_bits,23)
    sign_bit = bitwise_ops.bitwise_and(convert_x,SIGN_MASK)

    # round_up is the round value to add
    round_up = bitwise_ops.bitwise_or(exp_bits, sign_bit)

    final_x = bitwise_ops.bitwise_or(mantissa_bits, rest_bits)

    #print("final_x: ", final_x)
    x = tf.bitcast(final_x, tf.float32)
    round_up = tf.where(denormal_round, zeros_tensor, round_up)
    y = tf.bitcast(round_up, tf.float32)
    
    final_value = tf.where(half_bool, x + y, x)

    # filter out denormal from adding 
    final_value = tf.where(denormal, x, final_value)

    if num_mantissa == 0:
      final_value = tf.bitcast(rest_bits, tf.float32)

    t_value = final_value
  else:
    print("Unsupported Rounding Mode !!!!!!!!!!!!!!!!!!! ", round_mode)
  
  
  return t_value

def tf_fine_grain_custom_round(t_value, round_mode = "truncate", skip_processing = True, block_round_mode = "fine_grain_2_contiguous_exponent_reuse",
                            num_exp = 8, num_mantissa = 23, radix_exp = 2, block_size = 16, 
                            radix_mantissa = 2, is_tensor = True, transpose_tensors = True):
      
  # Fine grain breakup of numbers to create blocks 
  # need to traverse the tensor
  # convert tensor to numpy array
  
  # start = tf.timestamp()


  # TRANSPOSE_TENSORS = transpose_tensors
  
  # print("transpose_tensors: ", transpose_tensors)
  irregular_last_block = False
  non_tensor_input = False


  if len(t_value) > 1 and isinstance(t_value, list):
  # if len(t_value) > 1 :
  #          if not isinstance(layer.input, list):

    total_elements = 0
    new_concat_array = []
    shape_list = []
    size_list = []

    for i in range(len(t_value)):
      #new_concat_array = numpy.concatenate((new_concat_array, t_value[i]), axis=None)
      total_elements += tf.size(t_value[i])
      # calling transponse on tensor to create BFP blocks with values in depth wise dimension
      # TODO: May want to transpose with perm=[2,0,1], flatten, then block and check distribution
      # TODO: Need to figure out how to block in a tile manner 
      if (transpose_tensors): 
        t_value[i] = tf.transpose(t_value[i])

      shape_list.append(tf.shape(t_value[i]))
      if (i < len(t_value) - 1):
        if (i == 0):
          size_list.append(tf.size(t_value[i]))
        else:
          size_list.append(tf.size(t_value[i]) + size_list[-1])
    
    new_concat_array = numpy.concatenate(t_value, axis=None)
    new_concat_array = tf.cast(new_concat_array, tf.float32)

    # preprocessing code for list for numpy arrays 
    num_blocks = total_elements // block_size
    num_zeros = block_size - total_elements % block_size
    padded_zeros = (total_elements % block_size) != 0
    
    if (padded_zeros):
      zero_padding = tf.zeros([num_zeros], tf.float32)
      # Need to add code to pad last block with zeros if not full and remove the padding after rounding
      new_concat_array = numpy.concatenate((new_concat_array, zero_padding), axis=None )
      size_list.append(total_elements)
      num_blocks += 1

    flattened_tensor2 = tf.reshape(new_concat_array, [num_blocks, block_size])
    flattened_tensor2 = flattened_tensor2.numpy()

    # start2 = tf.timestamp()
    flattened_tensor2 = tf_custom_round(flattened_tensor2, skip_processing = skip_processing, round_mode = block_round_mode, block_size = block_size, num_mantissa=num_mantissa, is_tensor = False)
    # stop2 = tf.timestamp()
    # print("tf_custom_round exe Time = ", stop2 - start2)

    flattened_tensor2 = tf.reshape(flattened_tensor2, [-1])

    # post processing code here
    flattened_tensor2 = np.split(flattened_tensor2, size_list)

    if (padded_zeros):
      flattened_tensor2.pop()


    for i, arr in enumerate(flattened_tensor2):
      # print(len(arr))
      flattened_tensor2[i] = numpy.reshape(arr, shape_list[i])
      # Calling transpose to get back original shape
      if (transpose_tensors): 
        flattened_tensor2[i] = tf.transpose(flattened_tensor2[i])

    # flattened_tensor2 = numpy.reshape(split_tensor2, shape_list)

    t_value = flattened_tensor2    
      
  else:
    # Saving shape of tensor and getting total size

    # calling transponse on tensor to create BFP blocks with values in depth wise dimension 
    if (transpose_tensors): 
      t_value = tf.transpose(t_value)
    tensor_shape = tf.shape(t_value)#.numpy()
    total_elements = tf.size(t_value)#.numpy()

    # print("Tensor shape: ", tensor_shape)

    num_blocks = total_elements // block_size
    num_zeros = block_size - total_elements % block_size
    padded_zeros = (total_elements % block_size) != 0 

    if total_elements < block_size:
      if transpose_tensors:
        t_value = tf.transpose(t_value)
      return t_value

    if padded_zeros:
      zero_padding = tf.zeros([num_zeros], tf.float32)
      # print("total_elem, blocksize, num_zeros: ", total_elements, block_size, num_zeros)
      # print("t_value before padding:", t_value)
      # Need to add code to pad last block with zeros if not full and remove the padding after rounding
      t_value = numpy.concatenate((t_value, zero_padding), axis=None )
      # print("t_value after padding:", t_value)
      num_blocks += 1

    flattened_tensor2 = tf.reshape(t_value, [num_blocks, block_size])
    flattened_tensor2 = flattened_tensor2.numpy()

    # start2 = tf.timestamp()
    flattened_tensor2 = tf_custom_round(flattened_tensor2, skip_processing = skip_processing, round_mode = block_round_mode, block_size = block_size, num_mantissa=num_mantissa, is_tensor = False)
    # stop2 = tf.timestamp()
    # print("tf_custom_round exe Time = ", stop2 - start2)

    if (padded_zeros):
      # print("padded rounded tensor: ", flattened_tensor2)
      flattened_tensor2 = tf.reshape(flattened_tensor2, [-1])
      flattened_tensor2 = np.split(flattened_tensor2, [total_elements])
      flattened_tensor2.pop()    
      # print("unpadded rounded tensor: ", flattened_tensor2)

    t_value = tf.reshape(flattened_tensor2, tensor_shape)
    # Calling transpose to get back original shape
    if transpose_tensors:
      t_value = tf.transpose(t_value)

    # if (non_tensor_input):
    #   t_value = tf.RaggedTensor.from_tensor(t_value)

  # stop = tf.timestamp()
  # print("tf_fine_grain exe Time = ", stop - start)
  return t_value  
