nSpeakers =10;
nDims = 13; % dimensionality of feature vectors
nMixtures = 32; % How many mixtures used to generate data
nChannels = 10; % Number of channels (sessions) per speaker
nFrames = 1000; % Frames per speaker (10 seconds assuming 100 Hz)
nWorkers = 2; % Number of parfor workers, if available
rng('default'); % To promote reproducibility.
% Pick random centers for all the mixtures.
mixtureVariance = .10;
channelVariance = .05;
mixtureCenters = randn(nDims, nMixtures, nSpeakers);
channelCenters = randn(nDims, nMixtures, nSpeakers, nChannels)*.1;
trainSpeakerData = cell(nSpeakers, nChannels);
testSpeakerData = cell(nSpeakers, nChannels);
speakerID = zeros(nSpeakers, nChannels);
% Create the random data. Both training and testing data have the same
% layout.
for s=1:nSpeakers
    trainSpeechData = zeros(nDims, nMixtures);
    testSpeechData = zeros(nDims, nMixtures);
    for c=1:nChannels
        for m=1:nMixtures
            % Create data from mixture m for speaker s
            frameIndices = m:nMixtures:nFrames;
            nMixFrames = length(frameIndices);
            trainSpeechData(:,frameIndices) = ...
                randn(nDims, nMixFrames)*sqrt(mixtureVariance) + ...
                repmat(mixtureCenters(:,m,s),1,nMixFrames) + ...
                repmat(channelCenters(:,m,s,c),1,nMixFrames);
            testSpeechData(:,frameIndices) = ...
                randn(nDims, nMixFrames)*sqrt(mixtureVariance) + ...
                repmat(mixtureCenters(:,m,s),1,nMixFrames) + ...
                repmat(channelCenters(:,m,s,c),1,nMixFrames);
        end
        trainSpeakerData{s, c} = trainSpeechData;
        testSpeakerData{s, c} = testSpeechData;
        speakerID(s,c) = s; % Keep track of who this is
    end
end

