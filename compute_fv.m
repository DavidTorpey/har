function [X] = compute_fv(fold_num, fs, s, traj_means, traj_covariances, traj_priors, hog_means, hog_covariances, hog_priors, hof_means, hof_covariances, hof_priors, mbhx_means, mbhx_covariances, mbhx_priors, mbhy_means, mbhy_covariances, mbhy_priors)

    X = [];

    for i = 1:length(fs)
        fn = fs{i};
        disp(strcat('Fold ', num2str(fold_num), ': Computing FVs, Reading ', fn, ' (', num2str(i), ' of ', num2str(length(fs)), ')'));
        data = importdata(strcat('../iDT/', fn));
        
        traj = data(:, 1:30);
        hog = data(:, 31:126);
        hof = data(:, 127:234);
        mbhx = data(:, 235:330);
        mbhy = data(:, 331:end);
        
        traj_fv = vl_fisher(traj', traj_means, traj_covariances, traj_priors);
        hog_fv = vl_fisher(hog', hog_means, hog_covariances, hog_priors);
        hof_fv = vl_fisher(hof', hof_means, hof_covariances, hof_priors);
        mbhx_fv = vl_fisher(mbhx', mbhx_means, mbhx_covariances, mbhx_priors);
        mbhy_fv = vl_fisher(mbhy', mbhy_means, mbhy_covariances, mbhy_priors);
        
        encoding = [traj_fv; hog_fv; hof_fv; mbhx_fv; mbhy_fv];
        
        save(strcat('/scratch/dtorpey/UCFSports/FVs/loocv/fold_', num2str(fold_num), '/', s, '_', fn, '.mat'), 'encoding');
    end

end
