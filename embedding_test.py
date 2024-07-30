import whisperx

def transcribe_chunk(model, track1, track2):
  # it might make more sense to load the model once, globally, first. can refactor later
  if not model:
    device = 'cuda'
    batch_size = 16
    dtype = 'float16'
    model = whisperx.load_model("large-v2", device, compute_type=dtype, language='en')

  audio_1 = whisperx.load_audio(track1)
  audio_2 = whisperx.load_audio(track2)
  result1 = model.transcribe(audio_1, batch_size=batch_size)
  result2 = model.transcribe(audio_2, batch_size=batch_size)

  # align outputs
  model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
  result1 = whisperx.align(result1["segments"], model_a, metadata, audio_1, device, return_char_alignments=False)
  for line in result1['segments']:
    line['spkr'] = 'Speaker 1'

  result2 = whisperx.align(result2["segments"], model_a, metadata, audio_2, device, return_char_alignments=False)
  for line in result2['segments']:
    line['spkr'] = 'Speaker 2'

  conversation = result1['segments']
  conversation.extend(result2['segments'])
  formatted_output = []
  for line in sorted(conversation, key=lambda dic:dic['start']):
    formatted_output.append(f"{line['spkr']}: {line['text']}")
  return formatted_output,conversation

#def extract_first_instances(aligned_result):
  #extract timestamps for first instance of each speaker
  #first_instances = {'Speaker 1': (None, None), 'Speaker 2': (None, None)}
  #for segment in aligned_result:
    #speaker = segment['spkr']
    #if first_instances[speaker] == (None, None):
      #first_instances[speaker] = (segment['start'], segment['end'])

  #return first_instances
def get_first_two_instances(aligned_result, speaker_label):
  instances = []
  for segment in aligned_result:
    if segment['spkr'] == speaker_label and len(instances) < 5:
      instances.append((segment['start'], segment['end']))

  return instances




track1 = 'test.mp3'
track2 = 'test 2.mp3'
chunk_str = transcribe_chunk(None, track1,track2)
print(chunk_str[0])

timestamps = get_first_two_instances(chunk_str[1],"Speaker 2")
print(timestamps)

def speaker_test(timestamps,track1,track2):
    import torch
    from pyannote.audio import Model
    model = Model.from_pretrained("pyannote/embedding", 
                                use_auth_token="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")

    from pyannote.audio import Inference
    from pyannote.core import Segment
    inference = Inference(model, window="whole")
    excerpt1 = Segment(timestamps[0][0], timestamps[0][1])
    embedding1 = inference.crop("test.mp3", excerpt1)

    excerpt2 = Segment(timestamps[4][0], timestamps[4][1])
    embedding2 = inference.crop("test.mp3", excerpt2)

    import numpy as np
    from scipy.spatial.distance import cdist
    distance = cdist(np.reshape(embedding1, (1, -1)), np.reshape(embedding2, (1, -1)), metric="cosine")[0,0]
    print (distance) 
speaker_test(timestamps,track1,track2)