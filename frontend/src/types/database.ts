/**
 * Supabase Database Types
 * =======================
 * Type definitions for Supabase tables.
 * TODO: Generate from Supabase CLI: npx supabase gen types typescript
 */

export interface Database {
  public: {
    Tables: {
      workflows: {
        Row: {
          id: string;
          client_id: string;
          workflow_type: string;
          status: string;
          input_data: Record<string, unknown>;
          output_data: Record<string, unknown> | null;
          error_message: string | null;
          created_at: string;
          updated_at: string;
          completed_at: string | null;
        };
        Insert: Omit<Database['public']['Tables']['workflows']['Row'], 'id' | 'created_at' | 'updated_at'>;
        Update: Partial<Database['public']['Tables']['workflows']['Insert']>;
      };
      pending_approvals: {
        Row: {
          id: string;
          workflow_id: string;
          client_id: string;
          approval_type: string;
          title: string;
          description: string | null;
          data: Record<string, unknown>;
          status: 'pending' | 'approved' | 'rejected';
          created_at: string;
          updated_at: string;
          resolved_at: string | null;
          resolved_by: string | null;
        };
        Insert: Omit<Database['public']['Tables']['pending_approvals']['Row'], 'id' | 'created_at' | 'updated_at'>;
        Update: Partial<Database['public']['Tables']['pending_approvals']['Insert']>;
      };
      documents: {
        Row: {
          id: string;
          client_id: string;
          filename: string;
          content_type: string;
          size_bytes: number;
          storage_path: string;
          metadata: Record<string, unknown>;
          created_at: string;
        };
        Insert: Omit<Database['public']['Tables']['documents']['Row'], 'id' | 'created_at'>;
        Update: Partial<Database['public']['Tables']['documents']['Insert']>;
      };
    };
    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
  };
}
