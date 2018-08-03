%%
rng('default')
% Step1: Create the universal background model from all the
% training speaker data
nmix = nMixtures; % In this case, we know the # of mixtures needed
final_niter = 10;
ds_factor = 1;
ubm = gmm_em(trainSpeakerData(:), nmix, final_niter, ds_factor, ...
nWorkers);
%%
% Step2: Now adapt the UBM to each speaker to create GMM speaker model.
map_tau = 10.0;
config = 'mwv';
gmm = cell(nSpeakers, 1);
for s=1:nSpeakers
gmm{s} = mapAdapt(trainSpeakerData(s, :), ubm, map_tau, config);
end
%%
% Step3: Now calculate the score for each model versus each speaker's
% data.
% Generate a list that tests each model (first column) against all the
% testSpeakerData.
trials = zeros(nSpeakerstest*nChannelstest*nSpeakers, 2);
answers = zeros(nSpeakerstest*nChannelstest*nSpeakers, 1);
for ix = 1 : nSpeakers,
b = (ix-1)*nSpeakerstest*nChannelstest + 1;
e = b + nSpeakerstest*nChannelstest - 1;
trials(b:e, :) = [ix * ones(nSpeakerstest*nChannelstest, 1),(1:nSpeakerstest*nChannelstest)'];
answers((ix-1)*nChannelstest+b : (ix-1)*nChannelstest+b+nChannelstest-1) = 1;
end
gmmScores = score_gmm_trials(gmm, reshape(testSpeakerData', nSpeakerstest*nChannelstest,1), trials, ubm);
%%
% Step4: Now compute the EER and plot the DET curve and confusion matrix
% imagesc(reshape(gmmScores,nSpeakers*nChannelstest, nSpeakerstest))
% title('Speaker Verification Likelihood (GMM Model)');
% ylabel('Test # (Channel x Speaker)'); xlabel('Model #');
% colorbar; drawnow; axis xy
% %figure
% eer = compute_eer(gmmScores, answers, false);

gmmScores=reshape(gmmScores,nSpeakers*nChannelstest, nSpeakerstest);

%%
[val, idx] = max(gmmScores);
a=all_files(idx).name;
a=a(1:end-13);
sprintf('\n Identified Speaker is %s \n', a)
% fprintf('\n Identified Speaker is : S%i \n',idx)