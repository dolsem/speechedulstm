import wave, struct, csv, time
import pickle, numpy

root_path = ""
#this parses the transcript into a list of dictionaries for each time slice
def timeParse(i):
    with open(root_path+"TRN/SBC%03d.trn" % (i)) as f:
        name = ""
        result = []
        for line in f.readlines():
            try:
                foo = line.split("\t")
                if len(foo)==4:
                    start, end = map(float, foo[0:2])
                    name = foo[2].strip().strip(":") if not (foo[2].isspace() or len(foo[2])==0) else name
                elif len(foo) == 3:
                    start, end = map(float, foo[0].split())
                    name = foo[1].strip().strip(":") if not (foo[1].isspace() or len(foo[1])==0) else name
                else:
                    print "formatting transcript error: ",i, foo
                    continue
#                 print foo, start, end, name
                result +=[{"start":start, "end":end, "name":name.upper()}]
            except ValueError:
                print "Value Error: %s" % (line.strip())
                continue
        return result

#this parses the metadata files with the given headers and returns them as a dictionary
#name maps to row data as dictionary
def metaParse(i):
    result = {}
    dupes = set()
    with open(root_path+"meta/metadata%d.csv" % i, 'rU') as f:
        for row in csv.DictReader(f):
            if row.get('name').upper() in result.keys() or row.get('name').upper() in dupes:
                result.pop(row.get('name').upper())
                dupes.update(row.get('name').upper())
            else:
                result.update({row.get('name').upper() : row })
    return result

#this parses the audio .wav files with the "wave" library and the "struct" library to unpack the byte stream
#pulls all samples from in the selection size, storing it as a list of tuples from the channels
def audioParseSelection(i, start, end):
    w = wave.open(root_path+"wave/SBC%03d.wav" % (i), 'r')
    w.setpos(int(start * w.getframerate()))
    try:
        raw = [struct.unpack("%dh" % (w.getnchannels()), w.readframes(1)) for _ in range(int(w.getframerate() * (end - start)))]
        w.close()
        return {"left": [x[0] for x in raw], "right": [x[0] for x in raw]}, w.getframerate()
    except Exception as e:
        print "%s at %d from %.2f to %.2f" % (str(e), i, start, end)
        return [],0 
    
#this combines all three data sources together into individual dictionary elements
#based on the times slices, it then pulls in all available metadata (matching the names)
#and audio values from the wave (using the start and end seconds)
def combine (i, time, meta):
#     print "%d, %s: %f to %f" % (i, time.get("name"), time.get("start"), time.get("end"))
    audio, freq = audioParseSelection(i, time.get("start"), time.get("end"))
    if time.get("name") in meta.keys() and len(audio) > 0:
        time.update(meta.get(time.get("name")))
        time.update({"conv_id" : i, "freq": freq})
#         print time, str(audio)[:100]
        time.update({"wav" : audio, "nframes" : len(audio)})
        return time
    else:
        print "data not found:", time.get("name"), time.get("start"), time.get("end"), len(audio)
  
def pickleData(meta, r, step):
    for i in r:
        times = timeParse(i)
        print "i: %d, total: %d" % (i, len(times))
        data = [combine(i, dict(time), meta) for time in times]
        clean(i, step, data)

def test(foo):
    return numpy.mean(map(abs, foo)) >500

def clean(i, step, original):
    result = []
    for x in original:
        if x == None:
            continue
        start, end, freq, audio= x['start'], x['end'], x['freq'], x['wav']
        newAudio = {'left':[], 'right':[]}
        for t in range(0, int((end - start) * freq), int(step * freq)):
            tEnd = int(min(t+step*freq, (end - start) * freq))
            audioLclip, audioRclip = list(audio['left'][t:tEnd]), list(audio['right'][t:tEnd])
            if test(audioLclip) and test(audioRclip):
                newAudio['left'] +=audioLclip
                newAudio['right'] += audioRclip
                
        if len(newAudio['left']) > step*freq /2:
            newX = dict(x)
            newX.update({'wav': newAudio})
            result+=[newX]
            
    print "%d -> %d" % (len(original), len(result))
    fOut = open("Output/sbClean%d.p" % (i), 'wb')
    pickle.dump(result, fOut)
    fOut.close()

ranges = [range(1,15), range(15,31), range(31, 47), range(47, 61)]
 
seconds = 1.0
x = 1 # 1-4 available
 
print "start"
pickleData(metaParse(x), ranges[x-1], seconds)
print "all done"
