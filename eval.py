#!/bin/env python

import os,sys
import json

import tensorflow as tf

import tokenization
import data_lib
import albert_model

from albert import AlbertConfig, AlbertModel


model_dir = "./pretrained_models/"

spm_model_file="./spm/spm_sample.model"


"""
def get_model(albert_config, max_seq_length, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,loss_multiplier):
    #Returns keras fuctional model
    float_type = tf.float32
    hidden_dropout_prob = FLAGS.classifier_dropout # as per original code relased
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    albert_layer = AlbertModel(config=albert_config, float_type=float_type)

    pooled_output, _ = albert_layer(input_word_ids, input_mask, input_type_ids)

    albert_model = tf.keras.Model(inputs=[input_word_ids,input_mask,input_type_ids],
                                  outputs=[pooled_output])

    albert_model.load_weights(init_checkpoint)

"""

#def get_model(albert_config, max_seq_length, num_labels, init_checkpoint, learning_rate,
#                     num_train_steps, num_warmup_steps,loss_multiplier):

def get_model(albert_config, max_seq_length):
    float_type = tf.float32

    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    print(albert_config)
    albert_layer = AlbertModel(config=albert_config, float_type=float_type)

    pooled_output, _ = albert_layer(input_word_ids, input_mask, input_type_ids)


    albert_model = tf.keras.Model(inputs=[input_word_ids,input_mask,input_type_ids],
                                  outputs=[pooled_output])

    #albert_model.load("pretrained_model/tf2_model.h5")

    #albert_model.load_weights("pretrained_model/checkpoint")
    #albert_model.load_weights("pretrained_model/ctl_step_114.ckpt-3")
    #albert_model.load_weights("pretrained_model/tf2_model.h5")

    return albert_model


albert_config_file = "model_configs/base/config.json"

albert_config = AlbertConfig.from_json_file(albert_config_file)

meta_data_file_path = "data_prepro/train_meta_data"

"""
{
      "task_type": "albert_pretraining",
      "train_data_size": 307,
      "max_seq_length": 512,
      "max_predictions_per_seq": 20
}
"""

max_seq_length = 512
max_predictions_per_seq = 20

with tf.io.gfile.GFile(meta_data_file_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

    max_seq_length = input_meta_data['max_seq_length']
    max_predictions_per_seq = input_meta_data['max_predictions_per_seq']


pretrain_model, core_model = albert_model.pretrain_model(albert_config, max_seq_length, max_predictions_per_seq)

#model = get_model(albert_config, max_seq_length)
#print(model.summary())
print("### pretrained model ###")
print(pretrain_model.summary())

"""
### pretrained model ###
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_word_ids (InputLayer)     [(None, 512)]        0                                            
__________________________________________________________________________________________________
input_mask (InputLayer)         [(None, 512)]        0                                            
__________________________________________________________________________________________________
input_type_ids (InputLayer)     [(None, 512)]        0                                            
__________________________________________________________________________________________________
albert_model (AlbertModel)      ((None, 768), (None, 11683584    input_word_ids[0][0]             
                                                                 input_mask[0][0]                 
                                                                 input_type_ids[0][0]             
__________________________________________________________________________________________________
masked_lm_positions (InputLayer [(None, 20)]         0                                            
__________________________________________________________________________________________________
cls (ALBertPretrainLayer)       ((None, 30000), (Non 3970226     albert_model[0][0]               
                                                                 albert_model[0][1]               
                                                                 masked_lm_positions[0][0]        
__________________________________________________________________________________________________
masked_lm_ids (InputLayer)      [(None, 20)]         0                                            
__________________________________________________________________________________________________
masked_lm_weights (InputLayer)  [(None, 20)]         0                                            
__________________________________________________________________________________________________
next_sentence_labels (InputLaye [(None, 1)]          0                                            
__________________________________________________________________________________________________
al_bert_pretrain_loss_and_metri (None,)              0           cls[0][0]                        
                                                                 cls[0][1]                        
                                                                 masked_lm_ids[0][0]              
                                                                 masked_lm_weights[0][0]          
                                                                 next_sentence_labels[0][0]       
==================================================================================================
Total params: 11,813,810
Trainable params: 11,813,810
Non-trainable params: 0
__________________________________________________________________________________________________
None
"""


print("\n### core model ###")
print(core_model.summary())

"""
### core model ###
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_word_ids (InputLayer)     [(None, 512)]        0                                            
__________________________________________________________________________________________________
input_mask (InputLayer)         [(None, 512)]        0                                            
__________________________________________________________________________________________________
input_type_ids (InputLayer)     [(None, 512)]        0                                            
__________________________________________________________________________________________________
albert_model (AlbertModel)      ((None, 768), (None, 11683584    input_word_ids[0][0]             
                                                                 input_mask[0][0]                 
                                                                 input_type_ids[0][0]             
==================================================================================================
Total params: 11,683,584
Trainable params: 11,683,584
Non-trainable params: 0
__________________________________________________________________________________________________
None

"""

"""
pretrain_model, core_model = albert_model.pretrain_model(
    albert_config, max_seq_length, max_predictions_per_seq)
    
if FLAGS.init_checkpoint:
    logging.info(f"pre-trained weights loaded from {FLAGS.init_checkpoint}")
    pretrain_model.load_weights(FLAGS.init_checkpoint)
"""

#sys.exit(0)

core_model.load_weights("pretrained_model/tf2_model.h5")

print("core model weights loaded")


"""
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
if latest_checkpoint_file:
    logging.info(
          'Checkpoint file %s found and restoring from '
          'checkpoint', latest_checkpoint_file)
    checkpoint.restore(latest_checkpoint_file).expect_partial()
    logging.info('Loading from checkpoint file completed')
"""

checkpoint = tf.train.Checkpoint(model=pretrain_model)
latest_checkpoint_file = tf.train.latest_checkpoint("pretrained_model")
if latest_checkpoint_file:
    checkpoint.restore(latest_checkpoint_file).expect_partial()
    print('Loading from checkpoint file completed')

#latest_checkpoint_file = tf.train.latest_checkpoint("pretrained_model")
#if latest_checkpoint_file:


#pretrain_model.load_weights("pretrained_model/tf2_model.h5")
#pretrain_model.load_weights("pretrained_model")
#pretrain_model.load("pretrained_model/ctl_step_114.ckpt-3")
#pretrain_model.load_weights(latest)

#print("pre train model weights loaded")

#print("gau")
#loaded = tf.train.load_checkpoint(model_dir)

#print(loaded.summary())
#sys.exit(0)

#loaded = tf.saved_model.load( os.path.join(model_dir, "1") )
#loaded = tf.saved_model.load( os.path.join(model_dir, "1") )

tokenizer = tokenization.FullTokenizer(vocab_file=None,spm_model_file=spm_model_file, do_lower_case=True)

text_a = "the movie was not good"
example = data_lib.InputExample(guid=0, text_a=text_a, text_b=None, label=0)

print(example)

labels = [0, 1]
max_seq_length = 128

feature = data_lib.convert_single_example(ex_index=0, example=example, label_list=labels, max_seq_length=max_seq_length, tokenizer=tokenizer)

print(feature.__dict__)

test_input_word_ids =tf.convert_to_tensor([feature.input_ids], dtype=tf.int32, name='input_word_ids')
test_input_mask     =tf.convert_to_tensor([feature.input_mask], dtype=tf.int32, name='input_mask')
test_input_type_ids =tf.convert_to_tensor([feature.segment_ids], dtype=tf.int32, name='input_type_ids')


output = core_model( inputs=[test_input_mask,test_input_type_ids,test_input_word_ids])

print(type(output))
print(len(output))
print(output[0].shape)
print(output[1].shape)

print("###")

output2 = pretrain_model( inputs=[test_input_mask,test_input_type_ids,test_input_word_ids])

print(type(output2))


#(1, 768)
#(1, 128, 768)

#print(output[0])


#logit = model( input_mask=test_input_mask,input_type_ids=test_input_type_ids,input_word_ids=test_input_word_ids )

#logit = loaded.signatures["serving_default"]( input_mask=test_input_mask,input_type_ids=test_input_type_ids,input_word_ids=test_input_word_ids )

sys.exit(0)

pred = tf.argmax(logit['output'], axis=-1, output_type=tf.int32)
prob = tf.nn.softmax(logit['output'], axis=-1)

print(f'Prediction: {pred} Probabilities: {prob}')
