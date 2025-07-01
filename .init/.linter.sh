#!/bin/bash
cd /home/kavia/workspace/code-generation/bookqueryai-98730-98739/pdf_qa_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

