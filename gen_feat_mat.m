function [X_traj, X_hog, X_hof, X_mbhx, X_mbhy, traj_coeff, hog_coeff, hof_coeff, mbhx_coeff, mbhy_coeff] = gen_feat_mat(fold_num, fs)

    X_traj = [];
    X_hog = [];
    X_hof = [];
    X_mbhx = [];
    X_mbhy = [];

    for i = 1:length(fs)
        fn = fs{i};
        disp(strcat('Fold ', num2str(fold_num), ': Create Mats, Reading ', fn, ' (', num2str(i), ' of ', num2str(length(fs)), ')'));
        data = importdata(strcat('../iDT/', fn));
        X_s = sample_rows(data, 0.85);

        traj = X_s(:, 1:30);
        hog = rootsift(X_s(:, 31:126));
        hof = rootsift(X_s(:, 127:234));
        mbhx = rootsift(X_s(:, 235:330));
        mbhy = rootsift(X_s(:, 331:end));
        
        traj_coeff = perform_pca(traj, 15);
        hog_coeff = perform_pca(hog, 48);
        hof_coeff = perform_pca(hof, 54);
        mbhx_coeff = perform_pca(mbhx, 48);
        mbhy_coeff = perform_pca(mbhy, 48);
        
        traj = traj * traj_coeff;
        hog = hog * hog_coeff;
        hof = hof * hof_coeff;
        mbhx = mbhx * mbhx_coeff;
        mbhy = mbhy * mbhy_coeff;

        X_traj = [X_traj; traj];
        X_hog = [X_hog; hog];
        X_hof = [X_hof; hof];
        X_mbhx = [X_mbhx; mbhx];
        X_mbhy = [X_mbhy; mbhy];
    end

end
