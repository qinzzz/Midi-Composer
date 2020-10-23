# Midi Composer

**Under construction**

**For now, ignore midiparser.py, song_componets/.**


## About music data processing

### dataset

first try: use jazz dataset

- number of songs:
    
- size: 


### basic music knowledge & assumptions
1. a midi file represents a song. 
    
    It consists of multiple tracks. A track consists of multiple songs
    
2. 理解midi文件
    
    每个文件的track0为tempo map；存储meta message.
    ```
    <meta message key_signature key='Bb' time=0>
    <meta message smpte_offset frame_rate=25 hours=32 minutes=0 seconds=0 frames=0 sub_frames=0 time=0>
    <meta message track_name name='Fourth Avenue Theme' time=0>
    <meta message sequencer_specific data=(0, 0, 0, 106, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 42, 0, 0, 64, 39, 0, 0, 19, 29, 0, 0, 70, 24, 0, 0, 20, 3, 0, 0, 64, 8, 0, 0, 20, 29, 0, 0, 64, 6, 0, 0, 70, 42, 0, 0, 0, 0, 0, 0, 70, 4, 0, 0, 33, 5, 0, 0, 49, 42, 0, 0, 20, 39, 0, 0, 19, 29, 0, 0, 64, 6, 0, 0, 24, 42, 0, 0, 64, 39, 0, 0, 20, 29, 0, 0, 70, 24, 0, 0, 20, 27, 0, 0, 64, 8, 0, 0, 20, 1, 0, 0, 64, 24, 0, 0, 70, 11, 0, 0, 0, 0, 0, 0) time=0>
    ......
    <meta message set_tempo tempo=250000 time=0>
    <meta message time_signature numerator=4 denominator=4 clocks_per_click=24 notated_32nd_notes_per_beat=8 time=0>
    <meta message end_of_track time=0> 
    ```
    tempo: microseconds per beat
    
    key_signature: 
    
    time_signature: is important for defining the overall rhythm (e.g. whether the piece is a march or a waltz) 
    which is independent of the tempo, and indicates the position of the bar lines in the staff notation.
    
    比如这里是4/4拍，以1/4音符为一拍
    
    - Beats and ticks
    *A beat is the same as a quarter note*. 
    Beats are divided into **ticks**, the smallest unit of time in MIDI.
    
    - Note
    
        每个note的 message: pitch, velocity, and time.
        ```
           note_on channel=1 note=37 velocity=78 time=88
        ```
        
    - class Note 
        - note: == message.note. 音调，用数字表示，对应88个钢琴键 (0-88)
        - tick: Absolute nb of ticks from the beginning of the track
        - duration: 不太懂还在看
    
3. 区分速度（tempo）和节拍（meter）
    
    tempo 表示演奏时一拍的长度；
    
    meter 用n/m表示，以n分音符为一拍，每小节m拍，与tempo独立。
    
    - 歌曲的速度由两者共同决定的，比如计算每个小节的长度：
    
        ticks_per_bar = ticks_per_beat（与tempo有关）* beats_per_bar（与meter有关）
        
        e.g., If you are in 4/4 time and have 4 ticks per beat then you have 16 ticks per bar.
        
    - meter的表示：
    
        MAXIMUM_SONG_RESOLUTION = 4 表示最小的音符为1/4音符;
        NOTES_PER_BAR = 4 表示一小节有四个音符.
        
        以上两者表示此歌曲为4/4拍 (以四分音符为一拍，每小节四拍)。
    
    - tempo的表示：
    
        ticks_per_beat = 96
    
4. 如何表示一首歌

    **pinao roll**
    
    每首歌都可以用一个` [num_track, pitch_range, time] `矩阵来表示。
    
    其中 num_track 表示轨道数量；
    pitch_range 表示整个音域中的音符数（在这里用钢琴键数=88）；
    time 表示整首歌的最大长度。
    
    其中，最大分辨率设为1/32音符，即将time step坐标转化为以1/32音符的数量计数。`time = old_time * 32/4 / resolution`

5. *deprecated* 如何解决每首歌节奏不同的问题？
    
    不考虑tempo，归一化每个note的tick。
    
    scale：独立于tempo的时间单元。
    ```
    scale = 4 * self.ticks_per_beat // (Song.MAXIMUM_SONG_RESOLUTION * Song.NOTES_PER_BAR)
    note.tick(normalized) = note.tick * scale
    ```
   
    比如，对于4/4拍的曲子，如果每个小节note.tick=1, 则归一化后note.tick = 1 * (4 * 96 / 4 * 4) = 24;
    对于4/8拍的曲子，如果每个小节note.tick=1, 则归一化后note.tick = 1 * (4 * 96 / 8 * 4) = 12。
    
    这样note.tick就能准确反应每个音符的长短，与节拍无关。
    
## About generation model

one-hot编码由于数据中0的数量过多，预测结果偏差很大。

some ideas and code are borrowed from https://github.com/Conchylicultor/MusicGenerator