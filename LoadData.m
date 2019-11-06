function Data = LoadData(FolderName)
% This function uses the user given input Folder to load the dataset.
% LoadData considers all the variables after the first column as a double
% type format. NOTE: dirr function from Math File Exchange
% (https://www.mathworks.com/matlabcentral/fileexchange/
% 8682-dirr-find-files-recursively-filtering-name-date-or-bytes) is
% required prior to runing this function
% ####################
% INPUT: FolderName - Directory of folder
% 
% OUTPUT: Data - Struct type variable with respective fields (Table format) 
% for various datasets in the folder
% ####################

FileNames = dirr([FolderName '\*.csv']);

% Change filenames from struct format to string format
FileNames2 = ""; % Initialize variable
% VarNames = "";
for fileN = 1:size(FileNames,1)
    FileNames2(fileN,1) = string(FileNames(fileN).name);
    
    % Extract workspace variable names
    VarNames = split(FileNames2,'.',2);
    % Remove extensions
    VarNames(:,2) = [];
end

FileNames = FileNames2;

% Replace Characters that are invalid for matlab struct field name
VarNames = strrep(VarNames, '-','_');
Data = struct();

% Load Data
for fileN = 1:size(FileNames,1)
    Destination = FolderName + FileNames(fileN);
    
    % Confirm that all the variables are in 'double' type format
    opts = detectImportOptions(Destination);
    DataVars = vertcat(opts.VariableNames(:));
    VarsType = vertcat(opts.VariableTypes(:));
    
    isdouble = string(VarsType(2:end)) == 'double'; % 1st column Datetime is not changed
    
    % Change other formats to double expect the first\Datetime column
    if ~all(isdouble)
        opts = setvartype(opts,DataVars(2:end),'double');
    end
    
    % Check for detected datatypes as non-numeric (except Datetime Column)
    Data.(VarNames(fileN)) = readtable(Destination, opts); 
end