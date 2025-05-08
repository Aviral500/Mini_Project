import ffmpeg

stream = ffmpeg.input('test_audio.mp3')
stream = ffmpeg.output(stream, 'out.wav')
ffmpeg.run(stream)
