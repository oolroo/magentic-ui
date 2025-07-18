import React from "react";
import { useTranslation } from "react-i18next";

interface DocumentIframeProps {
  docUrl?: string;
  style?: React.CSSProperties;
  className?: string;
}

const DocumentIframe: React.FC<DocumentIframeProps> = ({
  docUrl,
  style,
  className,
}) => {
  const { t } = useTranslation();
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const [loading, setLoading] = React.useState<boolean>(false);
  const [error, setError] = React.useState<string | null>(null);
  console.log(docUrl);
  React.useEffect(() => {
    let isCancelled = false;
    const renderDocx = async () => {
      if (!docUrl || !containerRef.current) return;
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(docUrl);
        if (!response.ok) {
          throw new Error(`Failed to fetch document: ${response.status}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        if (isCancelled) return;

        const mod: any = await import("docx-preview");
        const docx = mod.default || mod;

        if (containerRef.current) {
          containerRef.current.innerHTML = "";
        }

        await docx.renderAsync(arrayBuffer, containerRef.current, undefined, {
          className: "docx-preview",
          inWrapper: true,
          ignoreWidth: false,
          ignoreHeight: false,
          breakPages: true,
        });
      } catch (e: any) {
        if (!isCancelled) {
          setError(e?.message || "Failed to render document");
        }
      } finally {
        if (!isCancelled) setLoading(false);
      }
    };

    renderDocx();

    return () => {
      isCancelled = true;
    };
  }, [docUrl]);

  if (!docUrl) {
    return (
      <div className="flex-1 w-full h-full min-h-0 flex items-center justify-center">
        <p>{t('Waiting for document to load...')}</p>
      </div>
    );
  }

  return (
    <div className={`flex-1 w-full h-full flex flex-col ${className || ''}`} style={style}>
      {loading && (
        <div className="w-full py-2 text-center text-gray-500">
          {t('Loading document...')}
        </div>
      )}
      {error && (
        <div className="w-full py-2 text-center text-red-600">
          {t('Failed to display document')}: {error}
        </div>
      )}
      <div
        ref={containerRef}
        className="flex-1 w-full overflow-auto bg-white"
        style={{ contain: "content" }}
      />
    </div>
  );
};

export default DocumentIframe;
