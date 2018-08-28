%%
rng('default');
% Step1: Create the universal background model from all the
% training speaker data
nmix = nMixtures;% In this case, we know the # of mixtures needed
final_niter = 10;
ds_factor = 1;
ubm = gmm_em(trainSpeakerData(:), nmix, final_niter, ...
ds_factor, nWorkers);
%%
% Step2.1: Calculate the statistics needed for the iVector model.
stats = cell(nSpeakers, nChannels);
for s=1:nSpeakers
for c=1:nChannels
[N,F] = compute_bw_stats(trainSpeakerData{s,c}, ubm);
stats{s,c} = [N; F];
end
end
% Step2.2: Learn the total variability subspace from all the speaker data.
tvDim = 100;
niter = 5;
T = train_tv_space(stats(:), ubm, tvDim, niter, nWorkers);
%
% Now compute the ivectors for each speaker and channel.
% The result is size
% tvDim x nSpeakers x nChannels
devIVs = zeros(tvDim, nSpeakers, nChannels);
for s=1:nSpeakers
for c=1:nChannels
    devIVs(:, s, c) = extract_ivector(stats{s, c}, ubm, T);
end
end
%%
% Step3.1: Now do LDA on the iVectors to find the dimensions that
% matter.
ldaDim = min(100, nSpeakers-1);
devIVbySpeaker = reshape(devIVs, tvDim, nSpeakers*nChannels);
[V,D] = lda(devIVbySpeaker, speakerID(:));
finalDevIVs = V(:, 1:ldaDim)' * devIVbySpeaker;
% Step3.2: Now train a Gaussian PLDA model with development
% i-vectors
nphi = ldaDim; % should be <= ldaDim
niter = 10;
pLDA = gplda_em(finalDevIVs, speakerID(:), nphi, niter);
%%
% Step4.1: OK now we have the channel and LDA models. Let's build
% actual speaker
% models. Normally we do that with new enrollment data, but now
% we'll just reuse the development set.
averageIVs = mean(devIVs, 3); % Average IVs across channels.
modelIVs = V(:, 1:ldaDim)' * averageIVs;
%%
% Step4.2: Now compute the ivectors for the test set
% and score the utterances against the models
testIVs = zeros(tvDim, nSpeakerstest, nChannelstest);
for s=1:nSpeakerstest
for c=1:nChannelstest
[N, F] = compute_bw_stats(testSpeakerData{s, c}, ubm);
testIVs(:, s, c) = extract_ivector([N; F], ubm, T);
end
end
testIVbySpeaker = reshape(permute(testIVs, [1 3 2]), ...
tvDim, nSpeakerstest*nChannelstest);
finalTestIVs = V(:, 1:ldaDim)' * testIVbySpeaker;
%%
% Step5: Now score the models with all the test data.
ivScores = score_gplda_trials(pLDA, modelIVs, finalTestIVs);
%imagesc(ivScores)
%title('Speaker Verification Likelihood (iVector Model)');
%xlabel('Test # (Channel x Speaker)'); ylabel('Model #');
%colorbar; axis xy; drawnow;
answers = zeros(nSpeakers*nChannels*nSpeakers, 1);
for ix = 1 : nSpeakers
b = (ix-1)*nSpeakers*nChannels + 1;
answers((ix-1)*nChannels+b : (ix-1)*nChannels+b+nChannels-1) = 1;
end
%ivScores = reshape(ivScores', nSpeakers*nChannels* nSpeakers, 1);
%figure;
%eer = compute_eer(ivScores, answers, false);
%%
[val, idx] = max(ivScores);
c=all_files(idx).name;
c=c(1:end-13);
sprintf('\n Identified Speaker is %s \n', c)
%fprintf('Identified Speaker is : S%i \n',idx)
    