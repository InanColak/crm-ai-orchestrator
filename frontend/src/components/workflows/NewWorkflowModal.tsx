'use client';

import { useState } from 'react';
import { X, Play, AlertCircle, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui';
import { cn } from '@/lib/utils';
import { useTriggerWorkflow } from '@/hooks';
import { WORKFLOW_TYPES, type WorkflowType, type WorkflowTypeDefinition } from '@/types/workflow';

interface NewWorkflowModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (workflowId: string) => void;
}

export function NewWorkflowModal({ isOpen, onClose, onSuccess }: NewWorkflowModalProps) {
  const [selectedType, setSelectedType] = useState<WorkflowTypeDefinition | null>(null);
  const [formData, setFormData] = useState<Record<string, string>>({});
  const [step, setStep] = useState<'select' | 'configure'>('select');

  const { trigger, isLoading, error } = useTriggerWorkflow();

  if (!isOpen) return null;

  const handleTypeSelect = (type: WorkflowTypeDefinition) => {
    if (!type.ready) return;
    setSelectedType(type);
    setFormData({});
    setStep('configure');
  };

  const handleBack = () => {
    setStep('select');
    setSelectedType(null);
    setFormData({});
  };

  const handleInputChange = (name: string, value: string) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async () => {
    if (!selectedType) return;

    // Build input data
    const inputData: Record<string, unknown> = {};
    selectedType.inputFields.forEach((field) => {
      const value = formData[field.name];
      if (value) {
        // Handle special cases
        if (field.name === 'participants') {
          inputData[field.name] = value.split(',').map((p) => p.trim());
        } else {
          inputData[field.name] = value;
        }
      }
    });

    try {
      const response = await trigger({
        workflow_type: selectedType.value,
        input_data: inputData,
      });

      onSuccess?.(response.workflow_id);
      handleClose();
    } catch {
      // Error is handled by the hook
    }
  };

  const handleClose = () => {
    setStep('select');
    setSelectedType(null);
    setFormData({});
    onClose();
  };

  const isFormValid = () => {
    if (!selectedType) return false;
    return selectedType.inputFields
      .filter((f) => f.required)
      .every((f) => formData[f.name]?.trim());
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={handleClose}
      />

      {/* Modal */}
      <div className="relative bg-gray-900 rounded-xl shadow-2xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-hidden border border-gray-700">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div>
            <h2 className="text-lg font-semibold text-white">
              {step === 'select' ? 'New Workflow' : selectedType?.label}
            </h2>
            <p className="text-sm text-gray-400">
              {step === 'select'
                ? 'Select a workflow type to get started'
                : selectedType?.description}
            </p>
          </div>
          <button
            onClick={handleClose}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          {step === 'select' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {WORKFLOW_TYPES.map((type) => (
                <button
                  key={type.value}
                  onClick={() => handleTypeSelect(type)}
                  disabled={!type.ready}
                  className={cn(
                    'p-4 rounded-lg border text-left transition-all',
                    type.ready
                      ? 'border-gray-600 hover:border-brand-500 hover:bg-gray-800 cursor-pointer'
                      : 'border-gray-700 bg-gray-800/50 cursor-not-allowed opacity-60'
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="font-medium text-white">{type.label}</p>
                      <p className="text-sm text-gray-400 mt-1">
                        {type.description}
                      </p>
                    </div>
                    {type.ready ? (
                      <CheckCircle className="h-5 w-5 text-green-500 shrink-0" />
                    ) : (
                      <span className="text-xs bg-yellow-900/50 text-yellow-400 px-2 py-1 rounded">
                        Coming Soon
                      </span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {selectedType?.inputFields.map((field) => (
                <div key={field.name}>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    {field.label}
                    {field.required && (
                      <span className="text-red-400 ml-1">*</span>
                    )}
                  </label>
                  {field.type === 'textarea' ? (
                    <textarea
                      value={formData[field.name] || ''}
                      onChange={(e) =>
                        handleInputChange(field.name, e.target.value)
                      }
                      placeholder={field.placeholder}
                      rows={4}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent resize-none"
                    />
                  ) : field.type === 'select' ? (
                    <select
                      value={formData[field.name] || ''}
                      onChange={(e) =>
                        handleInputChange(field.name, e.target.value)
                      }
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
                    >
                      <option value="">Select...</option>
                      {field.options?.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={formData[field.name] || ''}
                      onChange={(e) =>
                        handleInputChange(field.name, e.target.value)
                      }
                      placeholder={field.placeholder}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
                    />
                  )}
                </div>
              ))}

              {error && (
                <div className="flex items-center gap-2 p-3 bg-red-900/30 border border-red-700 rounded-lg">
                  <AlertCircle className="h-5 w-5 text-red-400 shrink-0" />
                  <p className="text-sm text-red-300">{error}</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-700 bg-gray-800/50">
          {step === 'configure' ? (
            <>
              <Button variant="ghost" onClick={handleBack}>
                Back
              </Button>
              <Button
                onClick={handleSubmit}
                loading={isLoading}
                disabled={!isFormValid()}
                icon={<Play className="h-4 w-4" />}
              >
                Start Workflow
              </Button>
            </>
          ) : (
            <>
              <div />
              <Button variant="secondary" onClick={handleClose}>
                Cancel
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
