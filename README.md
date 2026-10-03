# Ollama + Túnel no Google Colab                                                
                                                                                
                                                                                
                                                                                
Script que instala e configura, em um único comando:                            
                                                                                
                                                                                
                                                                                
1. Ollama (instalador oficial)                                                  
                                                                                
2. Servidor em `0.0.0.0:<porta>`                                                
                                                                                
3. Download do modelo (padrão: `qwen3:14b`)                                     
                                                                                
4. Túnel público gratuito do Pinggy (sem token)                                 
                                                                                
5. Impressão dos endpoints (nativo e compatível com OpenAI)                     
                                                                                
                                                                                
                                                                                
## Uso no Colab                                                                 
                                                                                
                                                                                
                                                                                
Um comando só (ajuste usuário/repo/branch):                                     
                                                                                
                                                                                
                                                                                
    !curl -fsSL                                                                 
https://raw.githubusercontent.com/SEU_USUARIO/ollama-colab/main/colab_setup_olla
ma.py -o colab_setup_ollama.py && python3 colab_setup_ollama.py                 
                                                                                
                                                                                
                                                                                
Com opções:                                                                     
                                                                                
                                                                                
                                                                                
    !python3 colab_setup_ollama.py --model qwen3:14b --port 11434               
                                                                                
    !python3 colab_setup_ollama.py --tunnel none                                
                                                                                
    !python3 colab_setup_ollama.py --stop                                       
                                                                                
                                                                                
                                                                                
Veja todas as opções:                                                           
                                                                                
                                                                                
                                                                                
    !python3 colab_setup_ollama.py --help                                       
                                           
