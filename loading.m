y=[];
fs=[];
file_directory = 'data\VoiceRecorder\train_files\'; %address of training audio files

all_files = dir([file_directory '*.wav']);
for i=1:length(all_files)
    [y_temp, fs_temp] = audioread(sprintf('%s%s', file_directory, all_files(i).name));
    y_temp = y_temp(1:1689600);% Taking minimum of all files to have same input length
    fs(i,1)=fs_temp;
    y(:,i)=y_temp;
end

%%
nSpeakers = 9; %number of speakers
nDims = 12; % dimensionality of feature vectors
nMixtures = 32; % How many mixtures used to generate data
nChannels = 10; % Number of channels (sessions) per speaker
nFrames = 1000; % Frames per speaker (10 seconds assuming 100 Hz)
nWorkers = 2; % Number of parfor workers, if available
rng('default'); % To promote reproducibility.
mfccs1=[];
for i=1:nSpeakers
%     display(i);
    mfccs1(:,:,i) = melcepst(y(:,i), fs(i));
end

mfccsdata=cell(nSpeakers,nChannels);

for j=1:nSpeakers
    for i=1:nChannels
        mfccsdata{j,i}=(mfccs1((i*(nFrames)-(nFrames-1)):i*nFrames,:,j))';
        speakerID(j,i) = j;
    end
end
trainSpeakerData=mfccsdata;

%%
test_directory = 'data\VoiceRecorder\test_files\';
test_files = dir([test_directory '*.wav']);
j = 1;%select a test file
[yt,fst(6)] = audioread(sprintf('%s%s', test_directory, test_files(j).name));
%audioread('C:\Users\Satyam\Documents\MATLAB\SOP\data\VoiceRecorder\Sugam Singla testing2.wav');
b=test_files(j).name;
b=b(1:end-12);
sprintf('correct label for input %s \n',b)
nSpeakerstest=1;
nChannelstest=1;

mfcct= melcepst(yt, fst(6));


testSpeakerData=cell(nSpeakerstest,nChannelstest);
for i=1:nSpeakerstest
    for j=1:nChannelstest
        testSpeakerData{i,j}=mfcct(1:1000,1:12)';
    end
end

