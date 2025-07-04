from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from datetime import datetime
import re

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))

def read_document():
    """Reads the content of the document.txt file"""
    try:
        with open('document.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print("File document.txt not found!")
        return None

def extract_entities_from_document(document_text):
    """Extracts structured entities from the document using regex and parsing"""
    entities = {}
    
    # Extract invoice number
    invoice_match = re.search(r'fattura numero (\d+)', document_text, re.IGNORECASE)
    entities['invoice_number'] = invoice_match.group(1) if invoice_match else None
    
    # Extract amount
    amount_match = re.search(r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*euro', document_text)
    entities['amount'] = amount_match.group(1).replace('.', '').replace(',', '.') if amount_match else None
    
    # Extract dates
    def to_iso_date(date_str):
        # Accepts both 15/03/2024 and 15-03-2024 and returns 2024-03-15
        if not date_str:
            return None
        parts = re.split(r'[/-]', date_str)
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        return date_str  # fallback
    
    issue_date_match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', document_text)
    entities['issue_date'] = to_iso_date(issue_date_match.group(1)) if issue_date_match else None
    
    due_date_match = re.search(r'scadenza.*?(\d{2}[/-]\d{2}[/-]\d{4})', document_text, re.IGNORECASE)
    entities['due_date'] = to_iso_date(due_date_match.group(1)) if due_date_match else None
    
    # Extract IBAN
    iban_match = re.search(r'(IT\d{2}[A-Z]\d{10}\d{12})', document_text)
    entities['iban'] = iban_match.group(1) if iban_match else None
    
    # Extract email
    email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', document_text)
    entities['email'] = email_match.group(1) if email_match else None
    
    # Extract phone
    phone_match = re.search(r'(\+39\s*\d{3}\s*\d{7})', document_text)
    entities['phone'] = phone_match.group(1) if phone_match else None
    
    # Extract tax code
    cf_match = re.search(r'([A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z])', document_text)
    entities['tax_code'] = cf_match.group(1) if cf_match else None
    
    return entities

def create_advanced_document_graph(tx, entities):
    """Creates an advanced and detailed graph based on the extracted entities"""
    
    # Delete any existing data
    tx.run("MATCH (n) DETACH DELETE n")
    
    # Convert amount to float if present
    amount_value = float(entities['amount']) if entities['amount'] else 2500.00
    
    # Create the graph with many more entities and semantic relationships
    tx.run("""
        // ===== ORGANIZATION ENTITY =====
        CREATE (acme:Organization:Company {
            name: 'ACME S.p.A.',
            legal_form: 'Società per Azioni',
            country: 'Italia',
            sector: 'Business',
            created_at: datetime(),
            status: 'Active'
        })
        
        CREATE (admin_dept:Department {
            name: 'Ufficio Amministrazione',
            type: 'Administrative',
            function: 'Billing and Finance',
            created_at: datetime()
        })
        
        // ===== PEOPLE ENTITY =====
        CREATE (laura:Person:Employee {
            name: 'Laura Bianchi',
            first_name: 'Laura',
            last_name: 'Bianchi',
            role: 'Referente Amministrativo',
            email: $email,
            phone: $phone,
            department: 'Amministrazione',
            status: 'Active',
            created_at: datetime()
        })
        
        CREATE (mario:Person:Customer {
            name: 'Mario Rossi',
            first_name: 'Mario',
            last_name: 'Rossi',
            tax_code: $tax_code,
            customer_type: 'Individual',
            status: 'Active',
            created_at: datetime()
        })
        
        // ===== DOCUMENT ENTITY =====
        CREATE (invoice:Document:Invoice {
            number: $invoice_number,
            type: 'Invoice',
            amount: $amount,
            currency: 'EUR',
            issue_date: date($issue_date_formatted),
            due_date: date($due_date_formatted),
            status: 'Issued',
            payment_status: 'Pending',
            language: 'Italian',
            created_at: datetime()
        })
        
        // ===== FINANCIAL ENTITIES =====
        CREATE (payment_method:PaymentMethod {
            type: 'Bank Transfer',
            description: 'Bonifico Bancario',
            currency: 'EUR',
            is_active: true
        })
        
        CREATE (bank_account:BankAccount:Asset {
            iban: $iban,
            country: 'IT',
            account_type: 'Business',
            status: 'Active',
            purpose: 'Receivables'
        })
        
        CREATE (transaction:FinancialTransaction {
            amount: $amount,
            currency: 'EUR',
            type: 'Invoice',
            status: 'Pending',
            due_date: date($due_date_formatted),
            created_at: datetime()
        })
        
        // ===== TEMPORAL ENTITIES =====
        CREATE (issue_event:Event:BusinessEvent {
            type: 'Invoice Issued',
            date: date($issue_date_formatted),
            description: 'Fattura emessa da ACME S.p.A.',
            created_at: datetime()
        })
        
        CREATE (due_event:Event:BusinessEvent {
            type: 'Payment Due',
            date: date($due_date_formatted),
            description: 'Scadenza pagamento fattura',
            created_at: datetime()
        })
        
        // ===== CATEGORICAL ENTITIES =====
        CREATE (business_process:Process {
            name: 'Invoice Management',
            type: 'Financial Process',
            stage: 'Active'
        })
        
        CREATE (communication:Communication {
            type: 'Business Correspondence',
            channel: 'Document',
            language: 'Italian',
            purpose: 'Payment Request'
        })
        
        // ===== ORGANIZATIONAL RELATIONSHIPS =====
        CREATE (laura)-[:WORKS_FOR]->(acme)
        CREATE (laura)-[:MEMBER_OF]->(admin_dept)
        CREATE (admin_dept)-[:PART_OF]->(acme)
        CREATE (mario)-[:CUSTOMER_OF]->(acme)
        
        // ===== DOCUMENT RELATIONSHIPS =====
        CREATE (acme)-[:ISSUES]->(invoice)
        CREATE (invoice)-[:ADDRESSED_TO]->(mario)
        CREATE (laura)-[:RESPONSIBLE_FOR]->(invoice)
        CREATE (invoice)-[:HANDLED_BY]->(admin_dept)
        
        // ===== FINANCIAL RELATIONSHIPS =====
        CREATE (invoice)-[:REQUIRES_PAYMENT_VIA]->(payment_method)
        CREATE (payment_method)-[:USES_ACCOUNT]->(bank_account)
        CREATE (bank_account)-[:OWNED_BY]->(acme)
        CREATE (invoice)-[:GENERATES]->(transaction)
        CREATE (transaction)-[:TO_ACCOUNT]->(bank_account)
        CREATE (mario)-[:OWES]->(transaction)
        
        // ===== TEMPORAL RELATIONSHIPS =====
        CREATE (invoice)-[:ISSUED_ON]->(issue_event)
        CREATE (invoice)-[:DUE_ON]->(due_event)
        CREATE (issue_event)-[:PRECEDES]->(due_event)
        
        // ===== PROCESS RELATIONSHIPS =====
        CREATE (invoice)-[:PART_OF_PROCESS]->(business_process)
        CREATE (business_process)-[:MANAGED_BY]->(admin_dept)
        CREATE (invoice)-[:COMMUNICATION_TYPE]->(communication)
        
        // ===== CONTACT RELATIONSHIPS =====
        CREATE (laura)-[:CONTACT_FOR]->(invoice)
        CREATE (mario)-[:CAN_CONTACT]->(laura)
    """, {
        'invoice_number': entities['invoice_number'],
        'amount': amount_value,
        'issue_date_formatted': entities['issue_date'] if entities['issue_date'] else '2024-03-15',
        'due_date_formatted': entities['due_date'] if entities['due_date'] else '2024-03-31',
        'iban': entities['iban'],
        'email': entities['email'],
        'phone': entities['phone'],
        'tax_code': entities['tax_code']
    })
    
    print("Advanced graph created successfully.")
    print(f"Entities created: {get_entity_counts(tx)}")

def get_entity_counts(tx):
    """Counts the entities in the graph"""
    result = tx.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
    counts = {}
    for record in result:
        for label in record['labels']:
            counts[label] = counts.get(label, 0) + record['count']
    return counts

def advanced_graph_analysis(tx):
    """Performs advanced analyses on the graph"""
    
    print("\n" + "="*60)
    print("Advanced document graph analysis")
    print("="*60)
    
    # 1. Structural analysis of the graph
    print("\n1. Graph structure:")
    result = tx.run("""
        MATCH (n)
        RETURN labels(n) as entity_types, count(n) as count
        ORDER BY count DESC
    """)
    
    total_nodes = 0
    for record in result:
        entity_type = '/'.join(record['entity_types'])
        count = record['count']
        total_nodes += count
        print(f"   • {entity_type}: {count} entities")
    
    # Count relationships
    rel_result = tx.run("MATCH ()-[r]->() RETURN count(r) as total_relationships")
    total_rels = rel_result.single()['total_relationships']
    print(f"   • Total nodes: {total_nodes}")
    print(f"   • Total relationships: {total_rels}")
    
    # 2. Critical business paths analysis
    print("\n2. Critical business paths:")
    result = tx.run("""
        MATCH path = (customer:Customer)-[*1..3]-(company:Company)
        RETURN path
    """)
    
    for record in result:
        print(f"   {record['path']}")
    
    # 3. Communication network analysis
    print("\n3. Communication network:")
    result = tx.run("""
        MATCH (customer:Customer)-[:CAN_CONTACT]->(employee:Employee)-[:RESPONSIBLE_FOR]->(invoice:Invoice)
        RETURN customer.name as cliente, employee.name as referente, 
               employee.email as email, employee.phone as telefono,
               invoice.number as fattura
    """)
    
    for record in result:
        print(f"   {record['cliente']} → {record['referente']}")
        print(f"      Email: {record['email']} | Phone: {record['telefono']}")
        print(f"      Invoice #{record['fattura']}")
    
    # 4. Timeline analysis
    print("\n4. Timeline of events:")
    result = tx.run("""
        MATCH (event:Event)-[:ISSUED_ON|DUE_ON]-(invoice:Invoice)
        RETURN event.type as tipo_evento, event.date as data, 
               event.description as descrizione, invoice.number as fattura
        ORDER BY event.date
    """)
    
    for record in result:
        print(f"   {record['data']}: {record['tipo_evento']}")
        print(f"      {record['descrizione']} (Invoice #{record['fattura']})")
    
    # 5. Financial analysis
    print("\n5. Financial analysis:")
    result = tx.run("""
        MATCH (transaction:FinancialTransaction)-[:TO_ACCOUNT]->(account:BankAccount)
        MATCH (customer:Customer)-[:OWES]->(transaction)
        RETURN customer.name as debitore, transaction.amount as importo,
               transaction.status as stato, account.iban as conto,
               transaction.due_date as scadenza
    """)
    
    for record in result:
        print(f"   Debtor: {record['debitore']}")
        print(f"      Amount: €{record['importo']} - Status: {record['stato']}")
        print(f"      IBAN: {record['conto']}")
        print(f"      Due date: {record['scadenza']}")
    
    # 6. Organizational dependencies analysis
    print("\n6. Organizational structure:")
    result = tx.run("""
        MATCH (dept:Department)-[:PART_OF]->(company:Company)
        MATCH (employee:Employee)-[:MEMBER_OF]->(dept)
        RETURN company.name as azienda, dept.name as dipartimento,
               collect(employee.name) as dipendenti, dept.function as funzione
    """)
    
    for record in result:
        print(f"   {record['azienda']}")
        print(f"      {record['dipartimento']} ({record['funzione']})")
        print(f"      Employees: {', '.join(record['dipendenti'])}")

def generate_business_insights(tx):
    """Generates business insights from the graph"""
    
    print("\n" + "="*60)
    print("Business insights")
    print("="*60)
    
    # Insight 1: Centrality Analysis
    result = tx.run("""
        MATCH (n)-[r]-()
        RETURN labels(n) as tipo, n.name as entita, count(r) as connessioni
        ORDER BY connessioni DESC
        LIMIT 5
    """)
    
    print("\nMost connected entities (centrality):")
    for record in result:
        print(f"   • {record['entita']} ({'/'.join(record['tipo'])}): {record['connessioni']} connections")
    
    # Insight 2: Risk Analysis
    result = tx.run("""
        MATCH (invoice:Invoice)-[:DUE_ON]->(due_event:Event)
        WHERE due_event.date < date()
        RETURN invoice.number as fattura_scaduta, invoice.amount as importo,
               due_event.date as scadenza
    """)
    
    overdue_invoices = list(result)
    if overdue_invoices:
        print("\nOverdue invoices:")
        for record in overdue_invoices:
            print(f"   • Invoice #{record['fattura_scaduta']}: €{record['importo']} (overdue on {record['scadenza']})")
    else:
        print("\nNo overdue invoices found")
    
    # Insight 3: Process Efficiency
    result = tx.run("""
        MATCH (process:Process)-[:MANAGED_BY]->(dept:Department)
        MATCH (invoice:Invoice)-[:PART_OF_PROCESS]->(process)
        RETURN process.name as processo, dept.name as dipartimento,
               count(invoice) as fatture_gestite
    """)
    
    print("\nProcess efficiency:")
    for record in result:
        print(f"   • {record['processo']} ({record['dipartimento']}): {record['fatture_gestite']} invoices")

def main():
    """Main function with complete error handling"""
    print("Advanced document analysis system\n")
    
    # Check configuration
    if not uri or not username or not password:
        print("Error: Missing environment variables!")
        print("Create a .env file with:")
        print("NEO4J_URI=neo4j://127.0.0.1:7687")
        print("NEO4J_USER=neo4j")
        print("NEO4J_PASSWORD=your_password")
        return
    
    # Read the document
    document_content = read_document()
    if not document_content:
        return
    
    print("Document content:")
    print("-" * 40)
    print(document_content)
    print("-" * 40)
    
    # Extract entities
    print("\nExtracting entities from the document...")
    entities = extract_entities_from_document(document_content)
    
    print("Extracted entities:")
    for key, value in entities.items():
        print(f"   • {key}: {value}")
    
    try:
        with driver.session() as session:
            # Create the advanced graph
            print(f"\nCreating advanced graph...")
            session.execute_write(create_advanced_document_graph, entities)
            
            # Perform analyses
            session.execute_read(advanced_graph_analysis)
            
            # Generate insights
            session.execute_read(generate_business_insights)
            
            print(f"\nAnalysis completed successfully.")
            print(f"Open Neo4j Browser: http://localhost:7474")
            print(f"Query to view everything: MATCH (n) RETURN n")
            
    except Exception as e:
        print(f"Error during operation: {e}")
        print("Suggestions:")
        print("   • Make sure Neo4j is running")
        print("   • Check credentials in the .env file")
    finally:
        driver.close()

if __name__ == "__main__":
    main()

# USEFUL QUERIES FOR NEO4J BROWSER:

# 1. View the entire graph
# MATCH (n) RETURN n

# 2. Show only main relationships
# MATCH (n)-[r]->(m) 
# WHERE n:Person OR n:Company OR n:Invoice
# RETURN n, r, m

# 3. Find paths between customer and company
# MATCH path = (customer:Customer)-[*1..3]-(company:Company)
# RETURN path

# 4. Timeline analysis
# MATCH (e:Event)
# RETURN e.type as event, e.date as date, e.description as description
# ORDER BY e.date

# 5. Graph schema
# CALL db.schema.visualization()