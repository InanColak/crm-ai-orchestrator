'use client';

import { useState, useCallback, useRef } from 'react';
import { Upload, X, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { isSupportedFileType, formatFileSize, ACCEPT_FILE_TYPES } from '@/types/document';

// =============================================================================
// FILE UPLOAD COMPONENT
// =============================================================================

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
  multiple?: boolean;
  maxSize?: number; // in bytes
  className?: string;
}

export function FileUpload({
  onFilesSelected,
  disabled = false,
  multiple = true,
  maxSize = 10 * 1024 * 1024, // 10MB default
  className,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFiles = useCallback(
    (files: File[]): { valid: File[]; errors: string[] } => {
      const valid: File[] = [];
      const errors: string[] = [];

      for (const file of files) {
        if (!isSupportedFileType(file.name)) {
          errors.push(`${file.name}: Unsupported file type`);
          continue;
        }

        if (file.size > maxSize) {
          errors.push(`${file.name}: File too large (max ${formatFileSize(maxSize)})`);
          continue;
        }

        valid.push(file);
      }

      return { valid, errors };
    },
    [maxSize]
  );

  const handleFiles = useCallback(
    (fileList: FileList | null) => {
      if (!fileList || fileList.length === 0) return;

      const files = Array.from(fileList);
      const { valid, errors } = validateFiles(files);

      if (errors.length > 0) {
        setError(errors.join('; '));
        // Clear error after 5 seconds
        setTimeout(() => setError(null), 5000);
      }

      if (valid.length > 0) {
        onFilesSelected(valid);
      }
    },
    [validateFiles, onFilesSelected]
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled) {
        setIsDragging(true);
      }
    },
    [disabled]
  );

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (disabled) return;

      handleFiles(e.dataTransfer.files);
    },
    [disabled, handleFiles]
  );

  const handleClick = useCallback(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.click();
    }
  }, [disabled]);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      handleFiles(e.target.files);
      // Reset input so the same file can be selected again
      if (inputRef.current) {
        inputRef.current.value = '';
      }
    },
    [handleFiles]
  );

  return (
    <div className={className}>
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          'relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all',
          isDragging
            ? 'border-brand-500 bg-brand-50'
            : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50',
          disabled && 'opacity-50 cursor-not-allowed',
          error && 'border-red-300 bg-red-50'
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_FILE_TYPES}
          multiple={multiple}
          onChange={handleInputChange}
          disabled={disabled}
          className="hidden"
        />

        <div className="flex flex-col items-center">
          <div
            className={cn(
              'h-12 w-12 rounded-full flex items-center justify-center mb-3',
              isDragging ? 'bg-brand-100' : 'bg-gray-100'
            )}
          >
            <Upload
              className={cn(
                'h-6 w-6',
                isDragging ? 'text-brand-600' : 'text-gray-400'
              )}
            />
          </div>

          <p className="text-sm font-medium text-gray-900">
            {isDragging ? 'Drop files here' : 'Drop files here or click to upload'}
          </p>

          <p className="mt-1 text-xs text-gray-500">
            PDF, DOCX, TXT, MD up to {formatFileSize(maxSize)}
          </p>
        </div>
      </div>

      {error && (
        <div className="mt-2 flex items-center gap-2 text-sm text-red-600">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// FILE UPLOAD LIST COMPONENT
// =============================================================================

interface UploadItem {
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress?: number;
  error?: string;
}

interface FileUploadListProps {
  items: UploadItem[];
  onRemove?: (file: File) => void;
}

export function FileUploadList({ items, onRemove }: FileUploadListProps) {
  if (items.length === 0) return null;

  return (
    <div className="space-y-2">
      {items.map((item, index) => (
        <FileUploadItem
          key={`${item.file.name}-${index}`}
          item={item}
          onRemove={onRemove}
        />
      ))}
    </div>
  );
}

interface FileUploadItemProps {
  item: UploadItem;
  onRemove?: (file: File) => void;
}

function FileUploadItem({ item, onRemove }: FileUploadItemProps) {
  const { file, status, error } = item;

  return (
    <div
      className={cn(
        'flex items-center gap-3 p-3 rounded-lg border',
        status === 'success' && 'bg-green-50 border-green-200',
        status === 'error' && 'bg-red-50 border-red-200',
        (status === 'pending' || status === 'uploading') && 'bg-gray-50 border-gray-200'
      )}
    >
      <div
        className={cn(
          'h-8 w-8 rounded flex items-center justify-center shrink-0',
          status === 'success' && 'bg-green-100',
          status === 'error' && 'bg-red-100',
          (status === 'pending' || status === 'uploading') && 'bg-gray-100'
        )}
      >
        {status === 'uploading' && (
          <Loader2 className="h-4 w-4 text-brand-600 animate-spin" />
        )}
        {status === 'success' && (
          <CheckCircle className="h-4 w-4 text-green-600" />
        )}
        {status === 'error' && (
          <AlertCircle className="h-4 w-4 text-red-600" />
        )}
        {status === 'pending' && (
          <FileText className="h-4 w-4 text-gray-400" />
        )}
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
        <p className="text-xs text-gray-500">
          {error || formatFileSize(file.size)}
        </p>
      </div>

      {onRemove && status !== 'uploading' && (
        <button
          onClick={() => onRemove(file)}
          className="p-1 rounded hover:bg-gray-200 transition-colors"
        >
          <X className="h-4 w-4 text-gray-400" />
        </button>
      )}
    </div>
  );
}
